use std::{collections::BTreeMap, sync::Arc, time::Duration};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};

use eframe::egui::CentralPanel;

use tokio::runtime::Runtime;

use crate::{
    dsp::{
        basic::{DebugNode, UiInput},
        AudioRate, ControlRate, Processor, Signal,
    },
    graph::{Connection, Graph, Node, NodeName},
    parser::{parse_script, DualGraphs},
    Scalar,
};

pub struct AudioContext {
    out_device: cpal::Device,
    stream: Option<cpal::Stream>,
}

impl AudioContext {
    fn write_data<T>(
        sample_rate: Scalar,
        graph: &Graph<AudioRate>,
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
        let mut binding = vec![Signal::new(0.0); output.len() / channels];
        out.insert(dac0, &mut binding);
        // }
        let ts = (0usize..(output.len() / channels))
            .map(|frame_idx| Signal::new(t as Scalar + frame_idx as Scalar / sample_rate))
            .collect::<Vec<_>>();
        let ins = BTreeMap::from_iter([(graph.node_id_by_name("t").unwrap(), &ts)]);
        graph.process_graph(sample_rate, &ins, &mut out);
        for (frame_idx, frame) in output.chunks_mut(channels).enumerate() {
            for (c, sample) in frame.iter_mut().enumerate() {
                *sample = T::from_sample(out[&dac0][frame_idx].value());
            }
        }
    }

    pub fn run<T>(&mut self, graph: Graph<AudioRate>, config: cpal::StreamConfig, buffer_len: usize)
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
                        &graph,
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
    audio_graph: Option<Graph<AudioRate>>,
    control_graph: Option<Graph<ControlRate>>,
    ui_control_inputs: Vec<Arc<Node<ControlRate>>>,
    rt: Option<Runtime>,
    sample_rate: Scalar,
    control_rate: Scalar,
    audio_buffer_len: usize,
}

impl PaprApp {
    pub fn new(sample_rate: Scalar, control_rate: Scalar, audio_buffer_len: usize) -> Self {
        Self {
            audio_cx: None,
            audio_graph: None,
            control_graph: None,
            ui_control_inputs: Vec::new(),
            rt: None,
            control_rate,
            sample_rate,
            audio_buffer_len,
        }
    }

    pub fn create_graphs(
        &mut self,
        audio_buffer_len: usize,
        sample_rate: Scalar,
        control_rate: Scalar,
    ) {
        let main_graphs = parse_script(
            include_str!("../test-scripts/test4.papr"),
            audio_buffer_len,
            sample_rate,
            control_rate,
        )
        .remove(&NodeName::new("main"))
        .unwrap();
        let DualGraphs {
            mut audio,
            name: _,
            mut control,
        } = main_graphs;

        for c_in_idx in control.input_node_indices.clone().iter().copied() {
            let c_in = &control.digraph[c_in_idx];
            if !c_in.inputs[0].implicit {
                let (an, cn) = UiInput::create_nodes(c_in.inputs[0].clone(), audio_buffer_len);
                self.ui_control_inputs.push(cn.clone());
                audio.add_node(an);
                let c_in_ui_idx = control.add_node(cn);
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

        for c_out_idx in control.output_node_indices.clone().iter().copied() {
            let c_out = &control.digraph[c_out_idx];
            let dbg_name = format!("{}_dbg", &c_out.name);
            let (_an, cn) = DebugNode::create_nodes(dbg_name.to_owned());
            let debug0_cn = control.add_node(cn);
            control.add_edge(
                c_out_idx,
                debug0_cn,
                Connection {
                    source_output: 0,
                    sink_input: 0,
                },
            );
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
        let rt = self
            .rt
            .take()
            .expect("PaprApp::spawn(): runtime not initialized");
        let mut audio_cx = self
            .audio_cx
            .take()
            .expect("PaprApp::spawn(): audio context not initialized");

        // let config = audio_cx.out_device.default_output_config().unwrap();
        let config = cpal::StreamConfig {
            channels: cpal::ChannelCount::from(1u16),
            sample_rate: cpal::SampleRate(self.sample_rate as u32),
            buffer_size: cpal::BufferSize::Fixed(self.audio_buffer_len as u32),
        };
        // let audio_buffer_len = if let SupportedBufferSize::Range { min, max } = config.buffer_size()
        println!("Output device: {}", audio_cx.out_device.name().unwrap());
        println!("Output config: {:?}", config);
        println!("Control rate: {} Hz", self.control_rate);

        self.create_graphs(
            self.audio_buffer_len,
            self.sample_rate as Scalar,
            self.control_rate as Scalar,
        );

        let audio_graph = self
            .audio_graph
            .take()
            .expect("PaprApp::spawn(): audio graph not initialized");
        let control_graph = self
            .control_graph
            .take()
            .expect("PaprApp::spawn(): control graph not initialized");

        audio_cx.run::<f32>(audio_graph, config, self.audio_buffer_len);
        self.audio_cx = Some(audio_cx);
        let t_idx = control_graph.node_id_by_name("t").unwrap();
        let control_rate = self.control_rate;
        std::thread::Builder::new()
            .name("PAPR Control".into())
            .spawn(move || {
                rt.block_on(async {
                    #[allow(clippy::unnecessary_cast)]
                    let mut clk = tokio::time::interval(Duration::from_secs_f64(
                        (control_rate as f64).recip(),
                    ));
                    let mut t = 0 as Scalar;
                    loop {
                        control_graph.process_graph(
                            control_rate,
                            &BTreeMap::from_iter([(t_idx, &vec![Signal::new(t)])]),
                            &mut BTreeMap::default(),
                        );
                        clk.tick().await;
                        t += control_rate.recip();
                    }
                });
            })
            .expect("PaprApp::spawn(): error spawning control rate thread");
    }
}

impl eframe::App for PaprApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        // ctx.request_repaint();

        CentralPanel::default().show(ctx, |ui| {
            for inp in self.ui_control_inputs.iter() {
                inp.processor.ui_update(ui);
            }
        });
    }
}
