use std::sync::{Arc, RwLock};

use crate::{
    graph::{Input, Node, NodeName, Output},
    Scalar,
};

use miette::Result;
use papr_proc_macro::node;

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
    ) -> Result<()> {
        outputs[0] = inputs[0].clone();
        Ok(())
    }
}

node! {
    pub struct GraphOutput;
    (input) -> (out)

    ~ {
        out = *input;
    }
}

node! {
    pub struct LetBinding;
    (input) -> (out)

    ~ {
        out = *input;
    }
}

node! {
    pub struct Var {
        value: Scalar,
    }
    (input, set) -> (out)

    ~ {
        if *set > 0.0 {
            self.value = *input;
        }
        out = self.value;
    }
}
