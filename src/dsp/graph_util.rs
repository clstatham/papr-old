use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::{
    dsp::{AudioRate, ControlRate},
    graph::{Input, InputName, Node, NodeName, Output, OutputName},
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
            NodeName(name.to_owned()),
            audio_buffer_len,
            FxHashMap::from_iter([(InputName::default(), for_input)]),
            FxHashMap::from_iter([(
                OutputName::default(),
                Output {
                    name: OutputName::default(),
                },
            )]),
            crate::graph::ProcessorType::Boxed(Box::new(Self)),
            None,
        ))
    }

    pub fn create_control_node(
        name: &str,
        for_input: Input<ControlRate>,
    ) -> Arc<Node<ControlRate>> {
        Arc::new(Node::new(
            NodeName(name.to_owned()),
            1,
            FxHashMap::from_iter([(InputName::default(), for_input)]),
            FxHashMap::from_iter([(
                OutputName::default(),
                Output {
                    name: OutputName::default(),
                },
            )]),
            crate::graph::ProcessorType::Boxed(Box::new(Self)),
            None,
        ))
    }
}

impl Processor<AudioRate> for GraphInput {
    fn process_sample(
        &self,
        buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() = inputs[&InputName::default()];
    }
}

impl Processor<ControlRate> for GraphInput {
    fn process_sample(
        &self,
        buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() = inputs[&InputName::default()];
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
        &self,
        buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() = inputs[&InputName::default()];
    }
}

impl Processor<ControlRate> for GraphOutput {
    fn process_sample(
        &self,
        buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() = inputs[&InputName::default()];
    }
}
