use std::sync::Arc;

use std::collections::BTreeMap;

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
            BTreeMap::from_iter([("input".to_owned(), for_input)]),
            BTreeMap::from_iter([(
                "out".to_owned(),
                Output {
                    name: "out".to_owned(),
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
            NodeName::new(name),
            1,
            BTreeMap::from_iter([("input".to_owned(), for_input)]),
            BTreeMap::from_iter([(
                "out".to_owned(),
                Output {
                    name: "out".to_owned(),
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
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &BTreeMap<&str, Signal<AudioRate>>,
        outputs: &mut BTreeMap<&str, Signal<AudioRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = inputs["input"];
    }
}

impl Processor<ControlRate> for GraphInput {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalRate>::SiblingNode>>,
        inputs: &BTreeMap<&str, Signal<ControlRate>>,
        outputs: &mut BTreeMap<&str, Signal<ControlRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = inputs["input"];
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
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &BTreeMap<&str, Signal<AudioRate>>,
        outputs: &mut BTreeMap<&str, Signal<AudioRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = inputs["input"];
    }
}

impl Processor<ControlRate> for GraphOutput {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalRate>::SiblingNode>>,
        inputs: &BTreeMap<&str, Signal<ControlRate>>,
        outputs: &mut BTreeMap<&str, Signal<ControlRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = inputs["input"];
    }
}
