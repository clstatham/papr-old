//! Based on https://github.com/SamiPerttu/fundsp/blob/ade89cc2a5f5cde5dfc5a723f8314fb0d44b4c89/src/oversample.rs

use std::sync::{Arc, RwLock};

use crate::{
    dsp::Signal,
    graph::{Node, ProcessorType},
};
use miette::Result;

use super::Processor;

pub struct Oversample2x {
    pub inner: Arc<Node>,
}

impl Oversample2x {
    pub fn new(inner: Arc<Node>) -> Self {
        Self { inner }
    }

    pub fn create_node(inner: Arc<Node>) -> Arc<Node> {
        let n = Node::new(
            format!("{}_oversampled", inner.name.clone()).into(),
            inner.inputs.clone(),
            inner.outputs.clone(),
            ProcessorType::Builtin(Box::new(RwLock::new(Self::new(inner)))),
        );
        Arc::new(n)
    }
}

impl Processor for Oversample2x {
    fn process_sample(
        &mut self,
        buffer_idx: usize,
        signal_rate: super::SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) -> Result<()> {
        // oversample by 2
        // todo: maybe keep buffer_len the same?
        let inner_sr = match signal_rate {
            super::SignalRate::Audio {
                sample_rate,
                buffer_len,
            } => super::SignalRate::Audio {
                sample_rate: sample_rate * 2.0,
                buffer_len: buffer_len * 2,
            },
            super::SignalRate::Control {
                sample_rate,
                buffer_len,
            } => super::SignalRate::Control {
                sample_rate: sample_rate * 2.0,
                buffer_len: buffer_len * 2,
            },
        };

        self.inner
            .processor
            .process_sample(buffer_idx * 2, inner_sr, inputs, outputs)?;

        self.inner
            .processor
            .process_sample(buffer_idx * 2 + 1, inner_sr, inputs, outputs)?;

        Ok(())
    }
}
