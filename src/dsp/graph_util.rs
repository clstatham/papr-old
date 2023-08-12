use std::sync::{Arc, RwLock};

use crate::{
    dsp::{AudioRate, ControlRate},
    graph::{Input, Node, NodeName, Output},
    node_constructor, Scalar,
};

use super::{Processor, Signal};

pub struct GraphInput;

impl GraphInput {
    pub fn create_audio_node(
        name: &str,
        audio_buffer_len: usize,
        for_input: Input<AudioRate>,
    ) -> Arc<Node<AudioRate>> {
        Arc::new(Node::new(
            NodeName::new(name),
            audio_buffer_len,
            vec![for_input],
            vec![Output {
                name: "out".to_owned(),
            }],
            crate::graph::ProcessorType::Builtin(Arc::new(RwLock::new(Self))),
            None,
        ))
    }

    pub fn create_control_node(
        name: &str,
        for_input: Input<ControlRate>,
    ) -> Arc<Node<ControlRate>> {
        Arc::new(Node::new(
            NodeName::new(name),
            1,
            vec![for_input],
            vec![Output {
                name: "out".to_owned(),
            }],
            crate::graph::ProcessorType::Builtin(Arc::new(RwLock::new(Self))),
            None,
        ))
    }
}

impl Processor<AudioRate> for GraphInput {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &[Signal<AudioRate>],
        outputs: &mut [Signal<AudioRate>],
    ) {
        *outputs.get_mut(0).unwrap() = inputs[0];
    }
}

impl Processor<ControlRate> for GraphInput {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalRate>::SiblingNode>>,
        inputs: &[Signal<ControlRate>],
        outputs: &mut [Signal<ControlRate>],
    ) {
        *outputs.get_mut(0).unwrap() = inputs[0];
    }
}

node_constructor! {
    pub struct GraphOutput;
    @in { input }
    @out { out }
    #in { input }
    #out { out }
}

impl Processor<AudioRate> for GraphOutput {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &[Signal<AudioRate>],
        outputs: &mut [Signal<AudioRate>],
    ) {
        *outputs.get_mut(0).unwrap() = inputs[0];
    }
}

impl Processor<ControlRate> for GraphOutput {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalRate>::SiblingNode>>,
        inputs: &[Signal<ControlRate>],
        outputs: &mut [Signal<ControlRate>],
    ) {
        *outputs.get_mut(0).unwrap() = inputs[0];
    }
}

node_constructor! {
    pub struct LetBinding;
    @in { input }
    @out { out }
    #in { input }
    #out { out }
}

impl Processor<AudioRate> for LetBinding {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &[Signal<AudioRate>],
        outputs: &mut [Signal<AudioRate>],
    ) {
        *outputs.get_mut(0).unwrap() = inputs[0];
    }
}

impl Processor<ControlRate> for LetBinding {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalRate>::SiblingNode>>,
        inputs: &[Signal<ControlRate>],
        outputs: &mut [Signal<ControlRate>],
    ) {
        *outputs.get_mut(0).unwrap() = inputs[0];
    }
}

node_constructor! {
    pub struct Var {
        value: Scalar,
    }
    @in { input, set }
    @out { out }
    #in { input, set }
    #out { out }
}

impl Processor<AudioRate> for Var {
    fn process_sample(
        &mut self,
        buffer_idx: usize,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<AudioRate as super::SignalRate>::SiblingNode>>,
        inputs: &[Signal<AudioRate>],
        outputs: &mut [Signal<AudioRate>],
    ) {
        if inputs[1].value() > 0.0 {
            self.value = inputs[0].value();
        }
        outputs[0] = self.value.into();
    }
}

impl Processor<ControlRate> for Var {
    fn process_sample(
        &mut self,
        buffer_idx: usize,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
        inputs: &[Signal<ControlRate>],
        outputs: &mut [Signal<ControlRate>],
    ) {
        if inputs[1].value() > 0.0 {
            self.value = inputs[0].value();
        }
        outputs[0] = self.value.into();
    }
}
