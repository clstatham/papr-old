use std::sync::{Arc, RwLock};

use crate::{
    graph::{Input, Node, NodeName, Output},
    Scalar,
};

use papr_proc_macro::node_constructor;

use super::{Processor, Signal, SignalRate};

pub struct GraphInput;

impl GraphInput {
    pub fn create_node(name: &str, for_input: Input) -> Arc<Node> {
        Arc::new(Node::new(
            NodeName::new(name),
            vec![for_input],
            vec![Output {
                name: "out".to_owned(),
            }],
            crate::graph::ProcessorType::Builtin(Box::new(RwLock::new(Self))),
        ))
    }
}

impl Processor for GraphInput {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = inputs[0];
    }
}

node_constructor! {
    pub struct GraphOutput;
    in { input }
    out { out }
}

impl Processor for GraphOutput {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = inputs[0];
    }
}

node_constructor! {
    pub struct LetBinding;
    in { input }
    out { out }
}

impl Processor for LetBinding {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = inputs[0];
    }
}

node_constructor! {
    pub struct Var {
        value: Scalar,
    }
    in { input, set }
    out { out }
}

impl Processor for Var {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        if inputs[1].value() > 0.0 {
            self.value = inputs[0].value();
        }
        outputs[0] = Signal::new(self.value);
    }
}
