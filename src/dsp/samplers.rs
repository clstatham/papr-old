use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};

use crate::{
    dsp::Signal,
    graph::{Input, Node, NodeName, Output},
    Scalar,
};

use super::{Processor, SignalRate};

pub struct Sample {
    pub buf: Box<[Scalar]>,
}

impl Sample {
    pub fn create_node(name: &str, audio_buffer_len: usize, sample_path: PathBuf) -> Arc<Node> {
        let dec = creak::Decoder::open(sample_path).unwrap();
        let channels = dec.info().channels();
        let buf: Box<[Scalar]> = dec
            .into_samples()
            .unwrap()
            .map(|s| s.unwrap_or(0.0) as Scalar)
            .collect::<Box<_>>()
            .chunks_exact(channels)
            .map(|ch| ch[0])
            .collect();
        Arc::new(Node::new(
            NodeName::new(name),
            audio_buffer_len,
            vec![Input::new("seek", Some(Signal::new(0.0)))],
            vec![
                Output {
                    name: "out".to_owned(),
                },
                Output {
                    name: "len".to_owned(),
                },
            ],
            crate::graph::ProcessorType::Builtin(Box::new(RwLock::new(Self { buf }))),
        ))
    }
}

impl Processor for Sample {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        let seek = inputs[0];
        let seek_samps = (seek.value() * signal_rate.rate()) as usize;

        outputs[0] = Signal::new(self.buf[seek_samps]);
        outputs[1] = Signal::new(self.buf.len() as Scalar / signal_rate.rate());
    }
}
