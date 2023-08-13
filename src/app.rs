use std::{
    collections::BTreeMap,
    env::current_dir,
    fs::File,
    io::Read,
    path::PathBuf,
    sync::Arc,
    thread::JoinHandle,
    time::{Duration, Instant},
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};

use eframe::egui::{CentralPanel, TopBottomPanel};

use tokio::runtime::Runtime;

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
        graph.process_graph(SignalRate::Audio, sample_rate, &ins, &mut out);
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
    rt: Option<Runtime>,
    sample_rate: Scalar,
    control_rate: Scalar,
    audio_buffer_len: usize,
    script_path: Option<PathBuf>,
    midi_ctx: Option<MidiContext>,
    control_thread_shutdown: Option<tokio::sync::oneshot::Sender<()>>,
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
            rt: None,
            control_rate,
            sample_rate,
            audio_buffer_len,
            script_path,
            midi_ctx: None,
            control_thread_shutdown: None,
        }
    }

    pub fn create_graphs(&mut self, audio_buffer_len: usize) {
        let path = self.script_path.as_ref().unwrap();
        let mut file = File::open(path).unwrap();
        let mut script = String::new();
        file.read_to_string(&mut script).unwrap();
        let (audio, mut control) =
            crate::parser2::parse_main_script(&script, &path.parent().unwrap(), audio_buffer_len)
                .unwrap();

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

        if self.rt.is_none() {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .enable_time()
                .build()
                .expect("PaprApp::init(): error creating new tokio runtime");
            self.rt = Some(rt);
        } else {
            eprintln!("PaprApp::init(): tokio runtime already initialized");
        }
    }

    pub fn spawn(&mut self) {
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

        self.create_graphs(self.audio_buffer_len);

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
                        SignalRate::Control,
                        control_rate,
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
    }
}

impl eframe::App for PaprApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        // ctx.request_repaint();
        TopBottomPanel::top("top_panel").show(ctx, |ui| {
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
                    self.spawn();
                }
            }
        });

        CentralPanel::default().show(ctx, |ui| {
            for inp in self.ui_control_inputs.iter() {
                inp.control_processor.ui_update(ui);
            }
        });
    }
}
