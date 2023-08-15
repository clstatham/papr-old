use nih_plug::prelude::*;
use nih_plug_egui::{create_egui_editor, egui, EguiState};
use papr::{
    app::PaprRuntime,
    dsp::{Signal, SignalRate},
    graph::Node,
    Scalar,
};
use papr_lib as papr;
use std::{
    collections::BTreeMap,
    env::current_dir,
    fs::File,
    io::Read,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Instant,
};

#[derive(Params)]
struct PaprParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
    // #[id = "gain"]
    // pub gain: FloatParam,
    // #[id = "freq"]
    // pub frequency: FloatParam,
    // #[id = "usemidi"]
    // pub use_midi: BoolParam,
}

impl Default for PaprParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(1600, 900),
        }
    }
}

pub struct Papr {
    params: Arc<PaprParams>,
    rt: Option<papr::app::PaprRuntime>,
    control_node_refs: Vec<Arc<Node>>,
    script_path: Arc<RwLock<Option<PathBuf>>>,
    script_text: Arc<RwLock<String>>,
    status_text: Arc<RwLock<String>>,
    reset_switch: Option<crossbeam_channel::Receiver<()>>,
    save_file_rx: Option<crossbeam_channel::Receiver<PathBuf>>,
    load_file_rx: Option<crossbeam_channel::Receiver<PathBuf>>,
    audio_t: Scalar,
    sample_rate: Scalar,
}

impl Default for Papr {
    fn default() -> Self {
        Self {
            params: Arc::new(PaprParams::default()),
            rt: None,
            control_node_refs: Vec::new(),
            script_path: Arc::new(RwLock::new(None)),
            script_text: Arc::new(RwLock::new(String::new())),
            status_text: Arc::new(RwLock::new(String::new())),
            reset_switch: None,
            save_file_rx: None,
            load_file_rx: None,
            audio_t: 0.0,
            sample_rate: 1.0,
        }
    }
}

impl Papr {
    fn reload(&mut self) {
        macro_rules! status_error {
            ($e:expr) => {
                if let Err(err) = $e {
                    *self.status_text.write().unwrap() = err;
                    return;
                }
            };
        }
        if let Some(rt) = self.rt.as_mut() {
            status_error!(rt.create_graphs(&self.script_text.read().unwrap()));
            self.control_node_refs.clear();
            let mut control_graph = rt.control_graph.take().unwrap();
            self.control_node_refs = control_graph
                .digraph
                .node_indices()
                .map(|id| control_graph.digraph[id].clone())
                .collect();
            let t_idx = control_graph.node_id_by_name("t").unwrap();
            let control_rate = rt.control_rate;
            let (tx, mut rx) = tokio::sync::oneshot::channel();
            rt.control_thread_shutdown = Some(tx);

            std::thread::Builder::new()
                .name("PAPR Control".into())
                .spawn(move || {
                    let clk = std::time::Duration::from_secs_f64(control_rate.recip());
                    let mut t = 0.0;
                    loop {
                        if rx.try_recv().is_ok() {
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
                .unwrap();
        }
        *self.status_text.write().unwrap() = "Runtime is running.".into();
    }
}

impl Plugin for Papr {
    const NAME: &'static str = "PAPR";

    const VENDOR: &'static str = "C-STATE";

    const URL: &'static str = "https://github.com/clstatham/papr";

    const EMAIL: &'static str = "";

    const VERSION: &'static str = "0.0.1";

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        // AudioIOLayout {
        //     main_input_channels: None,
        //     main_output_channels: NonZeroU32::new(2),
        //     ..AudioIOLayout::const_default()
        // },
        AudioIOLayout {
            main_input_channels: None,
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    type SysExMessage = ();

    type BackgroundTask = PaprBackgroundTask;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate as Scalar;
        let mut rt = PaprRuntime::new(buffer_config.sample_rate as Scalar, 1000.0, None, None);
        rt.init();
        self.rt = Some(rt);
        // self.reload();

        true
    }

    fn reset(&mut self) {}

    fn deactivate(&mut self) {
        if let Some(rt) = self.rt.as_mut() {
            if let Err(e) = rt.panic(None) {
                *self.status_text.write().unwrap() = e;
            }
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let buffer_len = buffer.samples();
        let mut next_event = context.next_event();

        if let Some(reset_switch) = self.reset_switch.as_ref() {
            if let Ok(()) = reset_switch.try_recv() {
                self.reload();
                return ProcessStatus::KeepAlive;
            }
        }
        if let Some(load_rx) = self.load_file_rx.as_ref() {
            if let Ok(path) = load_rx.try_recv() {
                let mut f = File::open(&path).unwrap();
                {
                    let mut buf = self.script_text.write().unwrap();
                    buf.clear();
                    f.read_to_string(&mut buf).unwrap();
                }
                *self.script_path.write().unwrap() = Some(path);
                self.reload();
                return ProcessStatus::KeepAlive;
            }
        }
        if let Some(save_rx) = self.save_file_rx.as_ref() {
            if let Ok(path) = save_rx.try_recv() {
                let mut f = File::create(path).unwrap();
                {
                    let buf = self.script_text.read().unwrap();
                    use std::io::Write;
                    write!(f, "{}", &buf).unwrap();
                }
                return ProcessStatus::KeepAlive;
            }
        }

        if let Some(rt) = self.rt.as_mut() {
            if let Some(graph) = rt.audio_graph.as_mut() {
                assert_eq!(buffer.channels(), 1);
                let mut output = buffer
                    .iter_samples()
                    .flatten()
                    .map(|s| Signal::new(*s as Scalar))
                    .collect::<Vec<_>>();
                let mut out = BTreeMap::new();
                let dac0 = graph
                    .node_id_by_name("dac0")
                    .expect("Expected `Main` graph to have at least `dac0` for outputs");
                out.insert(dac0, &mut output);

                let ts = (0usize..buffer_len)
                    .map(|frame_idx| {
                        Signal::new(self.audio_t as Scalar + frame_idx as Scalar / self.sample_rate)
                    })
                    .collect::<Vec<_>>();
                let ins = BTreeMap::from_iter([(graph.node_id_by_name("t").unwrap(), &ts)]);
                graph.process_graph(
                    SignalRate::Audio {
                        sample_rate: self.sample_rate,
                        buffer_len,
                    },
                    &ins,
                    &mut out,
                );

                for (frame_idx, mut frame) in buffer.iter_samples().enumerate() {
                    *frame.get_mut(0).unwrap() = out[&dac0][frame_idx].value() as f32;
                }
                self.audio_t += buffer_len as Scalar / self.sample_rate;
            }
        }

        ProcessStatus::KeepAlive
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let (reset_tx, rx) = crossbeam_channel::unbounded();
        self.reset_switch = Some(rx);
        let (save_tx, rx) = crossbeam_channel::unbounded();
        self.save_file_rx = Some(rx);
        let (load_tx, rx) = crossbeam_channel::unbounded();
        self.load_file_rx = Some(rx);
        create_egui_editor(
            self.params.editor_state.clone(),
            (
                self.script_path.clone(),
                self.script_text.clone(),
                self.status_text.clone(),
                self.control_node_refs.clone(),
            ),
            |_, _| {},
            move |ctx, _setter, state| {
                let (script_path, script_text, status_text, control_node_refs) = state;
                // ctx.request_repaint();

                egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
                    ui.with_layout(egui::Layout::left_to_right(egui::emath::Align::Min), |ui| {
                        if ui.button("Save").clicked() {
                            if let Some(script_path) = script_path.read().unwrap().clone() {
                                save_tx.send(script_path).unwrap();
                            } else if let Some(path) = rfd::FileDialog::new().save_file() {
                                // self.script_path = Some(path.clone());
                                save_tx.send(path.clone()).unwrap();
                            }
                        }
                        if ui.button("Save As...").clicked() {
                            if let Some(path) = rfd::FileDialog::new().save_file() {
                                save_tx.send(path).unwrap();
                            }
                        }
                        if ui.button("Open...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("PAPR Scripts", &["papr"])
                                .set_directory(
                                    script_path
                                        .read()
                                        .unwrap()
                                        .as_ref()
                                        .and_then(|p| p.parent())
                                        .unwrap_or(current_dir().unwrap().as_path()),
                                )
                                .pick_file()
                            {
                                // self.script_path = Some(path.clone());
                                load_tx.send(path).unwrap();
                            }
                        }
                        if ui.button("Reload").clicked() {
                            reset_tx.try_send(()).unwrap();
                        }
                    });
                });
                egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
                    ui.label(&*status_text.read().unwrap());
                });
                egui::SidePanel::left("node_ui").show(ctx, |ui| {
                    for node in control_node_refs.iter() {
                        node.processor.ui_update(ui);
                    }
                });
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.add_sized(
                        ui.available_size(),
                        egui::TextEdit::multiline(&mut *script_text.write().unwrap()).code_editor(),
                    );
                });
            },
        )
    }

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        Box::new(|task| match task {})
    }
}

pub enum PaprBackgroundTask {}

impl Vst3Plugin for Papr {
    const VST3_CLASS_ID: [u8; 16] = *b"PAPRAudioRuntime";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Instrument,
        Vst3SubCategory::Synth,
        Vst3SubCategory::Tools,
    ];
}

nih_export_vst3!(Papr);
