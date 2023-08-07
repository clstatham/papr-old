use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::{
    dsp::{AudioRate, ControlRate},
    graph::{InputName, Node, OutputName},
    node_constructor, Scalar,
};

use super::{Processor, Signal};

node_constructor! {
    GraphInput {}
    @in { "in" = 0.0 }
    @out { "out" }
    #in { "in" = 0.0 }
    #out { "out" }
}

impl Processor<AudioRate> for GraphInput {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&OutputName("out".to_owned())).unwrap() =
            inputs[&InputName("in".to_owned())];
    }
}

impl Processor<ControlRate> for GraphInput {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName("out".to_owned())).unwrap() =
            inputs[&InputName("in".to_owned())];
    }
}

node_constructor! {
    GraphOutput {}
    @in { "in" = 0.0 }
    @out { "out" }
    #in { "in" = 0.0 }
    #out { "out" }
}

impl Processor<AudioRate> for GraphOutput {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&OutputName("out".to_owned())).unwrap() =
            inputs[&InputName("in".to_owned())];
    }
}

impl Processor<ControlRate> for GraphOutput {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName("out".to_owned())).unwrap() =
            inputs[&InputName("in".to_owned())];
    }
}
