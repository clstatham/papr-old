use std::{
    collections::BTreeMap,
    env::current_dir,
    error::Error,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};

use eframe::{
    egui::{
        Button, CentralPanel, Context, Layout, RichText, SidePanel, TextEdit, TopBottomPanel,
        Window,
    },
    epaint::{text::LayoutJob, Color32, Stroke},
};

use syntect::parsing::SyntaxDefinition;

use crate::{
    dsp::{Signal, SignalRate},
    graph::{Graph, Node},
    io::midi::MidiContext,
    Scalar,
};

pub struct AudioContext {
    out_device: cpal::Device,
    stream: Option<cpal::Stream>,
}

impl AudioContext {
    fn write_data<T>(
        sample_rate: Scalar,
        graph: &mut Graph,
        output: &mut [T],
        channels: usize,
        t: Scalar,
    ) where
        T: SizedSample + FromSample<Scalar>,
    {
        let buffer_len = output.len() / channels;
        let mut out = BTreeMap::new();
        // for c in 0..channels {
        let dac0 = graph
            .node_id_by_name("dac0")
            .expect("Expected `Main` graph to have at least `dac0` for outputs");
        let mut dac0_vec = vec![Signal::new(0.0); buffer_len];
        out.insert(dac0, &mut dac0_vec);
        // }
        let ts = (0usize..buffer_len)
            .map(|frame_idx| Signal::new(t as Scalar + frame_idx as Scalar / sample_rate))
            .collect::<Vec<_>>();
        let ins = BTreeMap::from_iter([(graph.node_id_by_name("t").unwrap(), &ts)]);
        graph.process_graph(
            SignalRate::Audio {
                sample_rate,
                buffer_len,
            },
            &ins,
            &mut out,
        );
        for (frame_idx, frame) in output.chunks_mut(channels).enumerate() {
            for (_c, sample) in frame.iter_mut().enumerate() {
                *sample = T::from_sample(out[&dac0][frame_idx].value());
            }
        }
    }

    pub fn run<T>(&mut self, mut graph: Graph, config: cpal::StreamConfig) -> Result<(), String>
    where
        T: SizedSample + FromSample<Scalar>,
    {
        let mut sample_clock = 0 as Scalar;
        let err_fn = |err| eprintln!("Error occurred on stream: {err}");

        let channels = 1;
        let sample_rate = config.sample_rate.0 as Scalar;
        let stream = self
            .out_device
            .build_output_stream(
                &config,
                move |data: &mut [T], _info: &cpal::OutputCallbackInfo| {
                    sample_clock += (data.len() as Scalar / channels as Scalar) / sample_rate;
                    Self::write_data(
                        sample_rate as Scalar,
                        &mut graph,
                        data,
                        channels,
                        sample_clock,
                    );
                },
                err_fn,
                None,
            )
            .map_err(|e| format!("{:?}", e))?;
        stream.play().map_err(|e| format!("{:?}", e))?;
        self.stream = Some(stream);
        Ok(())
    }
}

pub struct PaprRuntime {
    pub audio_graph: Option<Graph>,
    pub control_graph: Option<Graph>,
    control_node_refs: Vec<Arc<Node>>,
    sample_rate: Scalar,
    pub control_rate: Scalar,
    pub control_thread_shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    status_text: String,
    out_file_name: Option<PathBuf>,
    run_for: Option<u64>,
}

impl PaprRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_rate: Scalar,
        control_rate: Scalar,
        out_file_name: Option<PathBuf>,
        run_for: Option<u64>,
    ) -> Self {
        Self {
            audio_graph: None,
            control_graph: None,
            control_node_refs: Vec::new(),
            control_rate,
            sample_rate,
            control_thread_shutdown: None,
            status_text: "Welcome to PAPR! Runtime is off.".into(),
            out_file_name,
            run_for,
        }
    }

    pub fn create_graphs(&mut self, script_path: &Path) -> Result<(), String> {
        let (audio, control) = crate::parser3::parse_main_script(script_path).map_err(|e| {
            eprintln!("{:?}", e);
            self.status_text = format!("{}", e);
            self.status_text.clone()
        })?;

        for c_in_idx in control.digraph.node_indices() {
            let c_in = &control.digraph[c_in_idx];
            self.control_node_refs.push(c_in.clone());
        }

        self.audio_graph = Some(audio);
        self.control_graph = Some(control);
        self.status_text = "Graph creation successful.".into();
        Ok(())
    }

    pub fn create_audio_context(#[cfg(target_os = "linux")] force_alsa: bool) -> AudioContext {
        // if self.audio_cx.is_none() && self.out_file_name.is_none() {
        #[cfg(target_os = "linux")]
        let host = if force_alsa {
            println!("Initializing ALSA host.");
            cpal::host_from_id(cpal::HostId::Alsa).expect("PaprApp::init(): no ALSA host available")
        } else {
            println!("Initializing JACK host.");
            cpal::host_from_id(cpal::HostId::Jack).expect("PaprApp::init(): no JACK host available")
        };

        #[cfg(target_os = "windows")]
        let host = cpal::host_from_id(cpal::HostId::Wasapi)
            .expect("PaprApp::init(): no WASAPI host available");

        let out_device = host
            .default_output_device()
            .expect("PaprApp::init(): failed to find output device");
        AudioContext {
            out_device,
            stream: None,
        }
        // self.audio_cx = Some(inner);
        // }
    }

    pub fn init(&mut self) {
        self.status_text = "Initialization successful.".into();
    }

    pub fn spawn(
        &mut self,
        script_path: &Path,
        mut audio_cx: Option<AudioContext>,
    ) -> Result<Option<AudioContext>, String> {
        self.create_graphs(script_path)?;

        let mut audio_graph = self
            .audio_graph
            .take()
            .expect("PaprApp::spawn(): audio graph not initialized");
        let mut control_graph = self
            .control_graph
            .take()
            .expect("PaprApp::spawn(): control graph not initialized");

        let t_idx: petgraph::stable_graph::NodeIndex = control_graph.node_id_by_name("t").unwrap();
        let control_rate = self.control_rate;
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        self.control_thread_shutdown = Some(tx);

        if let Some(out_file_name) = self.out_file_name.as_ref() {
            let run_for = self.run_for.unwrap();
            let run_for_secs = (run_for as f64) / 1000.0;
            let run_for_samples = (run_for_secs * self.sample_rate) as usize;
            let mut out_buf = vec![0.0; run_for_samples];
            let mut sample_idx = 0;
            let channels = 1;
            let mut t = 0.0;
            while sample_idx < run_for_samples {
                let output = &mut out_buf[sample_idx..sample_idx + 1]; // not using the audio buffer length for now
                let buffer_len = output.len() / channels;
                let mut out = BTreeMap::new();
                // for c in 0..channels {
                let dac0 = audio_graph
                    .node_id_by_name("dac0")
                    .expect("Expected `Main` graph to have at least `dac0` for outputs");
                let mut dac0_vec = vec![Signal::new(0.0); buffer_len];
                out.insert(dac0, &mut dac0_vec);
                // }
                let ts = (0usize..buffer_len)
                    .map(|frame_idx| {
                        Signal::new(t as Scalar + frame_idx as Scalar / self.sample_rate)
                    })
                    .collect::<Vec<_>>();
                let ins = BTreeMap::from_iter([(audio_graph.node_id_by_name("t").unwrap(), &ts)]);
                audio_graph.process_graph(
                    SignalRate::Audio {
                        sample_rate: self.sample_rate,
                        buffer_len,
                    },
                    &ins,
                    &mut out,
                );
                control_graph.process_graph(
                    SignalRate::Control {
                        sample_rate: self.sample_rate,
                        buffer_len,
                    },
                    &ins,
                    &mut BTreeMap::new(),
                );
                for (frame_idx, frame) in output.chunks_mut(channels).enumerate() {
                    for (_c, sample) in frame.iter_mut().enumerate() {
                        *sample = out[&dac0][frame_idx].value() as f32;
                    }
                }

                sample_idx += buffer_len;
                t += (buffer_len as Scalar) / self.sample_rate;
            }
            let out = &out_buf[..run_for_samples];
            let mut out_file = File::create(out_file_name).unwrap();
            wav::write(
                wav::Header::new(wav::WAV_FORMAT_IEEE_FLOAT, 1, self.sample_rate as u32, 32),
                &wav::BitDepth::ThirtyTwoFloat(out.to_vec()),
                &mut out_file,
            )
            .unwrap();
        } else {
            let mut audio_cx = audio_cx
                .take()
                .expect("PaprApp::spawn(): audio context not initialized");

            let config = cpal::StreamConfig {
                channels: cpal::ChannelCount::from(1u16),
                sample_rate: cpal::SampleRate(self.sample_rate as u32),
                buffer_size: cpal::BufferSize::Default,
            };
            println!(
                "Output device: {}",
                audio_cx
                    .out_device
                    .name()
                    .unwrap_or_else(|_| "(unknown)".to_string())
            );
            println!("Output config: {:?}", config);
            println!("Control rate: {} Hz", self.control_rate);

            audio_cx.run::<f32>(audio_graph, config)?;

            std::thread::Builder::new()
                .name("PAPR Control".into())
                .spawn(move || {
                    #[allow(clippy::unnecessary_cast)]
                    let clk = std::time::Duration::from_secs_f64((control_rate as f64).recip());
                    let mut t = 0 as Scalar;
                    loop {
                        if rx.try_recv().is_ok() {
                            println!("Shutting down control rate thread.");
                            break;
                        }
                        let tik = Instant::now();
                        control_graph.process_graph(
                            SignalRate::Control {
                                sample_rate: control_rate,
                                buffer_len: 1,
                            },
                            &BTreeMap::from_iter([(t_idx, &vec![Signal::new(t); 1])]),
                            &mut BTreeMap::default(),
                        );
                        let time = Instant::now() - tik;

                        if clk.as_secs_f64() > time.as_secs_f64() {
                            std::thread::sleep(clk - time);
                        }
                        t += clk.as_secs_f64() as Scalar;
                    }
                })
                .expect("PaprApp::spawn(): error spawning control rate thread");
        }

        self.status_text = "Runtime is running.".into();
        Ok(audio_cx)
    }

    pub fn reload(
        &mut self,
        script_path: &Path,
        audio_cx: Option<AudioContext>,
    ) -> Result<Option<AudioContext>, String> {
        if let Some(tx) = self.control_thread_shutdown.take() {
            tx.send(())
                .map_err(|_| "Error shutting down control rate thread".to_owned())?;
            std::thread::sleep(Duration::from_millis(10)); // paranoid sleep in between graph switches
        }
        self.control_node_refs.clear();
        self.init();
        // self.script_path = None;
        self.spawn(script_path, audio_cx)
    }

    pub fn panic(
        &mut self,
        mut audio_cx: Option<AudioContext>,
    ) -> Result<Option<AudioContext>, String> {
        if let Some(mut cx) = audio_cx.take() {
            if let Some(stream) = cx.stream.take() {
                drop(stream); // immediately stop playback
                audio_cx = Some(cx);
            }
        }
        if let Some(tx) = self.control_thread_shutdown.take() {
            tx.send(())
                .map_err(|_| "Error shutting down control rate thread".to_owned())?;
        }
        self.status_text = "Runtime stopped.".into();
        Ok(audio_cx)
    }
}

pub struct PaprApp {
    pub rt: PaprRuntime,
    allowed_to_close: bool,
    show_close_confirmation: bool,
    pub audio_cx: Option<AudioContext>,
    pub script_path: Option<PathBuf>,
    pub script_text: String,
    pub midi_cx: Option<MidiContext>,
}

impl PaprApp {
    pub fn new(rt: PaprRuntime) -> Self {
        Self {
            rt,
            allowed_to_close: false,
            show_close_confirmation: false,
            audio_cx: None,
            script_path: None,
            script_text: String::new(),
            midi_cx: None,
        }
    }

    pub fn init_audio(&mut self, #[cfg(target_os = "linux")] force_alsa: bool) {
        self.audio_cx = Some(PaprRuntime::create_audio_context(
            #[cfg(target_os = "linux")]
            force_alsa,
        ));
    }

    pub fn init_midi(&mut self) {
        self.midi_cx = Some(MidiContext::new("PAPR Midi In", None));
    }

    pub fn load_script_file(&mut self, script_path: &PathBuf) -> Result<(), Box<dyn Error>> {
        // let path = self.script_path.as_ref()?;
        let mut file = File::open(script_path)?;
        self.script_text = String::new();
        file.read_to_string(&mut self.script_text)?;
        self.script_path = Some(script_path.to_owned());
        Ok(())
    }

    pub fn save_script_file(&mut self, script_path: &PathBuf) -> Result<(), Box<dyn Error>> {
        use std::io::Write;
        let mut file = File::create(script_path)?;
        write!(file, "{}", &self.script_text)?;
        Ok(())
    }
}

impl eframe::App for PaprApp {
    fn on_close_event(&mut self) -> bool {
        self.show_close_confirmation = true;
        self.allowed_to_close
    }

    fn update(&mut self, ctx: &eframe::egui::Context, frame: &mut eframe::Frame) {
        ctx.request_repaint();
        if self.show_close_confirmation {
            Window::new("Save before exiting?")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            self.show_close_confirmation = false;
                        }
                        if ui.button("No").clicked() {
                            self.allowed_to_close = true;
                            self.show_close_confirmation = false;
                            frame.close();
                        }
                        if ui.button("Yes").clicked() {
                            if let Some(script_path) = self.script_path.clone() {
                                self.save_script_file(&script_path);
                            } else if let Some(path) = rfd::FileDialog::new().save_file() {
                                self.script_path = Some(path.clone());
                                self.save_script_file(&path);
                            }
                            self.allowed_to_close = true;
                            self.show_close_confirmation = false;
                            frame.close();
                        }
                    })
                });
        }
        // ctx.input(|i| {
        //     if i.key_pressed(Key::F5) {
        //         self.audio_cx = self
        //             .rt
        //             .reload(&self.script_text, self.audio_cx.take())
        //             .unwrap_or_else(|e| {
        //                 self.rt.status_text = format!("Failed to reload: {e}");
        //                 None
        //             });
        //     }
        //     if i.modifiers.ctrl {
        //         if i.key_pressed(Key::Enter) {
        //             self.audio_cx = self
        //                 .rt
        //                 .reload(&self.script_text, self.audio_cx.take())
        //                 .unwrap_or_else(|e| {
        //                     self.rt.status_text = format!("Failed to reload: {e}");
        //                     None
        //                 });
        //         }
        //         if i.key_pressed(Key::S) {
        //             if let Some(script_path) = self.script_path.clone() {
        //                 self.save_script_file(&script_path);
        //             } else if let Some(path) = rfd::FileDialog::new().save_file() {
        //                 self.script_path = Some(path.clone());
        //                 self.save_script_file(&path);
        //             }
        //         }
        //         if i.key_pressed(Key::K) {
        //             self.audio_cx = self
        //                 .rt
        //                 .panic(self.audio_cx.take())
        //                 .expect("Error while panicking");
        //         }
        //     }
        // });
        frame.set_window_title(&format!(
            "PAPR - {}",
            self.script_path
                .as_ref()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("(untitled)")
        ));
        TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .add_sized(
                        (80.0, 40.0),
                        Button::new(RichText::new("PANIC").color(Color32::BLACK).size(20.0))
                            .fill(Color32::YELLOW)
                            .stroke(Stroke::new(1.0, Color32::BLACK)),
                    )
                    .clicked()
                {
                    self.audio_cx = self
                        .rt
                        .panic(self.audio_cx.take())
                        .expect("Error while panicking");
                }
                ui.label(&self.rt.status_text);
            });
        });
        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.with_layout(Layout::left_to_right(eframe::emath::Align::Min), |ui| {
                if ui.button("Save").clicked() {
                    if let Some(script_path) = self.script_path.clone() {
                        self.save_script_file(&script_path);
                    } else if let Some(path) = rfd::FileDialog::new().save_file() {
                        self.script_path = Some(path.clone());
                        self.save_script_file(&path);
                    }
                }
                if ui.button("Save As...").clicked() {
                    if let Some(path) = rfd::FileDialog::new().save_file() {
                        self.script_path = Some(path.clone());
                        self.save_script_file(&path);
                    }
                }
                if ui.button("Open...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("PAPR Scripts", &["papr"])
                        .set_directory(
                            self.script_path
                                .as_ref()
                                .and_then(|p| p.parent())
                                .unwrap_or(current_dir().unwrap().as_path()),
                        )
                        .pick_file()
                    {
                        self.script_path = Some(path.clone());
                        if let Some(tx) = self.rt.control_thread_shutdown.take() {
                            tx.send(()).unwrap();
                            std::thread::sleep(Duration::from_millis(10)); // paranoid sleep in between graph switches
                        }
                        self.rt.control_node_refs.clear();
                        self.init_audio(
                            #[cfg(target_os = "linux")]
                            false, // todo
                        );
                        self.init_midi();
                        self.rt.init();
                        self.load_script_file(&path).unwrap_or_else(|e| {
                            self.rt.status_text = format!("Failed to load script: {e}");
                        });

                        // self.audio_cx = Some(PaprRuntime::create_audio_context(false));

                        self.audio_cx =
                            self.rt
                                .spawn(&path, self.audio_cx.take())
                                .unwrap_or_else(|e| {
                                    self.rt.status_text = format!("Failed to spawn: {e}");
                                    None
                                });
                    }
                }
                if ui.button("Reload").clicked() {
                    self.audio_cx = self
                        .rt
                        .reload(self.script_path.as_ref().unwrap(), self.audio_cx.take())
                        .unwrap_or_else(|e| {
                            self.rt.status_text = format!("Failed to reload: {e}");
                            None
                        });
                }
            });
        });

        SidePanel::left("inputs").show(ctx, |ui| {
            for node in self.rt.control_node_refs.iter() {
                node.processor.ui_update(ui);
            }
        });

        CentralPanel::default().show(ctx, |ui| {
            let mut layouter = |ui: &eframe::egui::Ui, string: &str, _| {
                let layout_job = highlight(ui.ctx(), string);
                ui.fonts().layout_job(layout_job)
            };
            ui.add_sized(
                ui.available_size(),
                TextEdit::multiline(&mut self.script_text)
                    .font(eframe::egui::TextStyle::Monospace)
                    .code_editor()
                    .layouter(&mut layouter),
            );
        });
    }
}

/*
   Syntax highlighting impl based on:
   https://github.com/emilk/egui/blob/1023f937a67f3771e25c45cc49067acb71740e97/crates/egui_demo_lib/src/syntax_highlighting.rs
*/

pub fn highlight(ctx: &Context, code: &str) -> LayoutJob {
    impl eframe::egui::util::cache::ComputerMut<&str, LayoutJob> for Highlighter {
        fn compute(&mut self, key: &str) -> LayoutJob {
            self.highlight(key)
        }
    }

    type HighlightCache = eframe::egui::util::cache::FrameCache<LayoutJob, Highlighter>;

    ctx.memory().caches.cache::<HighlightCache>().get(code)
}

struct Highlighter {
    ps: syntect::parsing::SyntaxSet,
    ts: syntect::highlighting::ThemeSet,
}

impl Default for Highlighter {
    fn default() -> Self {
        let mut builder = syntect::parsing::SyntaxSetBuilder::new();
        builder.add(
            SyntaxDefinition::load_from_str(include_str!("../papr.sublime-syntax"), false, None)
                .expect("PAPR Sublime syntax file not found"),
        );
        let ps = builder.build();
        Self {
            ps,
            ts: syntect::highlighting::ThemeSet::load_defaults(),
        }
    }
}

impl Highlighter {
    fn highlight(&self, code: &str) -> LayoutJob {
        use syntect::easy::HighlightLines;
        use syntect::highlighting::FontStyle;
        use syntect::util::LinesWithEndings;

        let syntax = self
            .ps
            .find_syntax_by_name("PAPR")
            .expect("PAPR syntax not found");

        let theme = "base16-mocha.dark";
        let mut h = HighlightLines::new(syntax, &self.ts.themes[theme]);

        use eframe::egui::text::{LayoutSection, TextFormat};

        let mut job = LayoutJob {
            text: code.into(),
            ..Default::default()
        };

        for line in LinesWithEndings::from(code) {
            for (style, range) in h.highlight_line(line, &self.ps).unwrap() {
                let fg = style.foreground;
                let text_color = eframe::egui::Color32::from_rgb(fg.r, fg.g, fg.b);
                let italics = style.font_style.contains(FontStyle::ITALIC);
                let underline = style.font_style.contains(FontStyle::UNDERLINE);
                let underline = if underline {
                    eframe::egui::Stroke::new(1.0, text_color)
                } else {
                    eframe::egui::Stroke::none()
                };
                job.sections.push(LayoutSection {
                    leading_space: 0.0,
                    byte_range: as_byte_range(code, range),
                    format: TextFormat {
                        font_id: eframe::egui::FontId::monospace(16.0),
                        color: text_color,
                        italics,
                        underline,
                        ..Default::default()
                    },
                });
            }
        }

        job
    }
}

fn as_byte_range(whole: &str, range: &str) -> std::ops::Range<usize> {
    let whole_start = whole.as_ptr() as usize;
    let range_start = range.as_ptr() as usize;
    assert!(whole_start <= range_start);
    assert!(range_start + range.len() <= whole_start + whole.len());
    let offset = range_start - whole_start;
    offset..(offset + range.len())
}
