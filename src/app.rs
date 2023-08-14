use std::{
    collections::BTreeMap,
    env::current_dir,
    fs::File,
    io::Read,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};

use eframe::{
    egui::{
        Button, CentralPanel, Context, Key, Layout, RichText, SidePanel, TextEdit, TopBottomPanel,
        Window,
    },
    epaint::{text::LayoutJob, Color32, Stroke},
};

use syntect::parsing::SyntaxDefinition;

use crate::{
    dsp::{basic::UiInput, Signal, SignalRate},
    graph::{Connection, Graph, Node},
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
        buffer_len: usize,
        t: Scalar,
    ) where
        T: SizedSample + FromSample<Scalar>,
    {
        if output.len() != buffer_len {
            panic!("Buffer len mismatch: {} vs {}", output.len(), buffer_len);
        }
        let mut out = BTreeMap::new();
        // for c in 0..channels {
        let dac0 = graph.node_id_by_name("dac0").unwrap();
        let mut dac0_vec = vec![Signal::new(0.0); output.len() / channels];
        out.insert(dac0, &mut dac0_vec);
        // }
        let ts = (0usize..(output.len() / channels))
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

    pub fn run<T>(&mut self, mut graph: Graph, config: cpal::StreamConfig, buffer_len: usize)
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
                        buffer_len,
                        sample_clock,
                    );
                },
                err_fn,
                None,
            )
            .unwrap();
        stream.play().unwrap();
        self.stream = Some(stream);
    }
}

pub struct PaprApp {
    audio_cx: Option<AudioContext>,
    audio_graph: Option<Graph>,
    control_graph: Option<Graph>,
    ui_control_inputs: Vec<Arc<Node>>,
    sample_rate: Scalar,
    control_rate: Scalar,
    audio_buffer_len: usize,
    script_path: Option<PathBuf>,
    script_text: String,
    midi_ctx: Option<MidiContext>,
    control_thread_shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    allowed_to_close: bool,
    show_close_confirmation: bool,
    status_text: String,
}

impl PaprApp {
    pub fn new(
        script_path: Option<PathBuf>,
        sample_rate: Scalar,
        control_rate: Scalar,
        audio_buffer_len: usize,
    ) -> Self {
        Self {
            audio_cx: None,
            audio_graph: None,
            control_graph: None,
            ui_control_inputs: Vec::new(),
            control_rate,
            sample_rate,
            audio_buffer_len,
            script_path,
            script_text: String::new(),
            midi_ctx: None,
            control_thread_shutdown: None,
            allowed_to_close: false,
            show_close_confirmation: false,
            status_text: "Welcome to PAPR! Runtime is off.".into(),
        }
    }

    pub fn load_script_file(&mut self) -> PathBuf {
        let path = self.script_path.as_ref().unwrap();
        let mut file = File::open(path).unwrap();
        self.script_text = String::new();
        file.read_to_string(&mut self.script_text).unwrap();
        path.clone()
    }

    pub fn save_script_file(&mut self) {
        use std::io::Write;
        let path = self.script_path.as_ref().unwrap();
        let mut file = File::create(path).unwrap();
        write!(file, "{}", &self.script_text).unwrap();
    }

    pub fn create_graphs(&mut self, audio_buffer_len: usize) -> Result<(), String> {
        let (audio, mut control) = crate::parser2::parse_main_script(
            &self.script_text,
            &self
                .script_path
                .as_ref()
                .map(|p| p.parent().unwrap())
                .unwrap_or(current_dir().unwrap().as_path()),
            audio_buffer_len,
        )?;

        for c_in_idx in control.input_node_indices.clone().iter().copied() {
            let c_in = &control.digraph[c_in_idx];
            if !c_in.inputs[0].implicit {
                let node = UiInput::create_node(c_in.inputs[0].clone(), audio_buffer_len);
                self.ui_control_inputs.push(node.clone());
                let c_in_ui_idx = control.add_node(node);
                control.add_edge(
                    c_in_ui_idx,
                    c_in_idx,
                    Connection {
                        source_output: 0,
                        sink_input: 0,
                    },
                );
            }
        }

        self.audio_graph = Some(audio);
        self.control_graph = Some(control);
        self.status_text = "Graphs creation successful.".into();
        Ok(())
    }

    pub fn init(&mut self) {
        if self.audio_cx.is_none() {
            let host = cpal::host_from_id(cpal::HostId::Jack)
                .expect("PaprApp::init(): no JACK host available");
            let out_device = host
                .default_output_device()
                .expect("PaprApp::init(): failed to find output device");
            let inner = AudioContext {
                out_device,
                stream: None,
            };
            self.audio_cx = Some(inner);
        } else {
            eprintln!("PaprApp::init(): audio context already initialized");
        }

        if self.midi_ctx.is_none() {
            self.midi_ctx = Some(MidiContext::new("PAPR Midi In"));
        } else {
            eprintln!("PaprApp::init(): midi context already initialized");
        }

        self.status_text = "Initialization successful.".into();
    }

    pub fn spawn(&mut self) -> Result<(), String> {
        let mut audio_cx = self
            .audio_cx
            .take()
            .expect("PaprApp::spawn(): audio context not initialized");

        let config = cpal::StreamConfig {
            channels: cpal::ChannelCount::from(1u16),
            sample_rate: cpal::SampleRate(self.sample_rate as u32),
            buffer_size: cpal::BufferSize::Fixed(self.audio_buffer_len as u32),
        };
        println!("Output device: {}", audio_cx.out_device.name().unwrap());
        println!("Output config: {:?}", config);
        println!("Control rate: {} Hz", self.control_rate);

        self.create_graphs(self.audio_buffer_len)?;

        let audio_graph = self
            .audio_graph
            .take()
            .expect("PaprApp::spawn(): audio graph not initialized");
        let mut control_graph = self
            .control_graph
            .take()
            .expect("PaprApp::spawn(): control graph not initialized");

        audio_cx.run::<f32>(audio_graph, config, self.audio_buffer_len);
        self.audio_cx = Some(audio_cx);
        let t_idx = control_graph.node_id_by_name("t").unwrap();
        let control_rate = self.control_rate;
        let buffer_len = self.audio_buffer_len;
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        self.control_thread_shutdown = Some(tx);
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
                            buffer_len,
                        },
                        &BTreeMap::from_iter([(t_idx, &vec![Signal::new(t); buffer_len])]),
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
        self.status_text = "Runtime is running.".into();
        Ok(())
    }

    fn reload(&mut self) -> Result<(), String> {
        if let Some(tx) = self.control_thread_shutdown.take() {
            tx.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(10)); // paranoid sleep in between graph switches
        }
        self.ui_control_inputs.clear();
        self.init();
        // self.script_path = None;
        self.spawn()
    }

    fn panic(&mut self) {
        if let Some(mut cx) = self.audio_cx.take() {
            if let Some(stream) = cx.stream.take() {
                drop(stream); // immediately stop playback
            }
        }
        if let Some(tx) = self.control_thread_shutdown.take() {
            tx.send(()).unwrap();
        }
        self.status_text = "Runtime stopped.".into();
    }
}

impl eframe::App for PaprApp {
    fn on_close_event(&mut self) -> bool {
        self.show_close_confirmation = true;
        self.allowed_to_close
    }

    fn update(&mut self, ctx: &eframe::egui::Context, frame: &mut eframe::Frame) {
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
                            if self.script_path.is_some() {
                                self.save_script_file();
                            } else if let Some(path) = rfd::FileDialog::new().save_file() {
                                self.script_path = Some(path.clone());
                                self.save_script_file();
                            }
                            self.allowed_to_close = true;
                            self.show_close_confirmation = false;
                            frame.close();
                        }
                    })
                });
        }
        ctx.input(|i| {
            if i.key_pressed(Key::F5) {
                self.reload()
                    .unwrap_or_else(|e| self.status_text = format!("Failed to reload: {e}"));
            }
            if i.modifiers.ctrl {
                if i.key_pressed(Key::Enter) {
                    self.reload()
                        .unwrap_or_else(|e| self.status_text = format!("Failed to reload: {e}"));
                }
                if i.key_pressed(Key::S) {
                    if self.script_path.is_some() {
                        self.save_script_file();
                    } else if let Some(path) = rfd::FileDialog::new().save_file() {
                        self.script_path = Some(path.clone());
                        self.save_script_file();
                    }
                }
                if i.key_pressed(Key::K) {
                    self.panic();
                }
            }
        });
        frame.set_window_title(&format!(
            "PAPR - {}",
            self.script_path
                .as_ref()
                .map(|p| p.file_name().unwrap().to_str().unwrap())
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
                    self.panic();
                }
                ui.label(&self.status_text);
            });
        });
        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.with_layout(Layout::left_to_right(eframe::emath::Align::Min), |ui| {
                if ui.button("Save").clicked() {
                    if self.script_path.is_some() {
                        self.save_script_file();
                    } else if let Some(path) = rfd::FileDialog::new().save_file() {
                        self.script_path = Some(path.clone());
                        self.save_script_file();
                    }
                }
                if ui.button("Save As...").clicked() {
                    if let Some(path) = rfd::FileDialog::new().save_file() {
                        self.script_path = Some(path.clone());
                        self.save_script_file();
                    }
                }
                if ui.button("Open...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("PAPR Scripts", &["papr"])
                        .set_directory(
                            self.script_path
                                .as_ref()
                                .map(|p| p.parent().unwrap())
                                .unwrap_or(current_dir().unwrap().as_path()),
                        )
                        .pick_file()
                    {
                        self.script_path = Some(path);
                        if let Some(tx) = self.control_thread_shutdown.take() {
                            tx.send(()).unwrap();
                            std::thread::sleep(Duration::from_millis(10)); // paranoid sleep in between graph switches
                        }
                        self.ui_control_inputs.clear();
                        self.init();
                        self.load_script_file();
                        self.spawn()
                            .unwrap_or_else(|e| self.status_text = format!("Failed to spawn: {e}"));
                    }
                }
                if ui.button("Reload").clicked() {
                    self.reload()
                        .unwrap_or_else(|e| self.status_text = format!("Failed to reload: {e}"));
                }
            });
        });

        SidePanel::left("inputs").show(ctx, |ui| {
            for inp in self.ui_control_inputs.iter() {
                inp.processor.ui_update(ui);
            }
        });

        CentralPanel::default().show(ctx, |ui| {
            let mut layouter = |ui: &eframe::egui::Ui, string: &str, _| {
                let layout_job = highlight(ui.ctx(), string);
                ui.fonts(|f| f.layout_job(layout_job))
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

    ctx.memory_mut(|mem| mem.caches.cache::<HighlightCache>().get(code))
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
                .unwrap(),
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

        let syntax = self.ps.find_syntax_by_name("PAPR").unwrap();

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
                    eframe::egui::Stroke::NONE
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
