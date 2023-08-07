use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::{
    graph::{AudioRate, ControlRate, CreateNodes, Input, InputName, Node, Output, OutputName},
    Scalar,
};

use super::{Processor, Signal};

pub struct GraphInput;
pub struct GraphInputA;
pub struct GraphInputC;

impl Processor<AudioRate> for GraphInputA {
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

impl Processor<ControlRate> for GraphInputC {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::graph::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName("out".to_owned())).unwrap() =
            inputs[&InputName("in".to_owned())];
    }
}

impl CreateNodes for GraphInput {
    fn create_nodes() -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let cn = Arc::new(Node::new(
            FxHashMap::from_iter(
                [(
                    InputName("in".to_owned()),
                    Input {
                        name: InputName("in".to_owned()),
                        default: Signal::new_control(0.0),
                    },
                )]
                .into_iter(),
            ),
            FxHashMap::from_iter(
                [(
                    OutputName("out".to_owned()),
                    Output {
                        name: OutputName("out".to_owned()),
                    },
                )]
                .into_iter(),
            ),
            Box::new(GraphInputC).into(),
            None,
        ));
        let an = Arc::new(Node::new(
            FxHashMap::from_iter(
                [(
                    InputName("in".to_owned()),
                    Input {
                        name: InputName("in".to_owned()),
                        default: Signal::new_audio(0.0),
                    },
                )]
                .into_iter(),
            ),
            FxHashMap::from_iter(
                [(
                    OutputName("out".to_owned()),
                    Output {
                        name: OutputName("out".to_owned()),
                    },
                )]
                .into_iter(),
            ),
            Box::new(GraphInputA).into(),
            Some(cn.clone()),
        ));
        (an, cn)
    }
}

pub struct GraphOutput;
pub struct GraphOutputA;
pub struct GraphOutputC;

impl Processor<AudioRate> for GraphOutputA {
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

impl Processor<ControlRate> for GraphOutputC {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::graph::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName("out".to_owned())).unwrap() =
            inputs[&InputName("in".to_owned())];
    }
}

impl CreateNodes for GraphOutput {
    fn create_nodes() -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let cn = Arc::new(Node::new(
            FxHashMap::from_iter(
                [(
                    InputName("in".to_owned()),
                    Input {
                        name: InputName("in".to_owned()),
                        default: Signal::new_control(0.0),
                    },
                )]
                .into_iter(),
            ),
            FxHashMap::from_iter(
                [(
                    OutputName("out".to_owned()),
                    Output {
                        name: OutputName("out".to_owned()),
                    },
                )]
                .into_iter(),
            ),
            Box::new(GraphOutputC).into(),
            None,
        ));
        let an = Arc::new(Node::new(
            FxHashMap::from_iter(
                [(
                    InputName("in".to_owned()),
                    Input {
                        name: InputName("in".to_owned()),
                        default: Signal::new_audio(0.0),
                    },
                )]
                .into_iter(),
            ),
            FxHashMap::from_iter(
                [(
                    OutputName("out".to_owned()),
                    Output {
                        name: OutputName("out".to_owned()),
                    },
                )]
                .into_iter(),
            ),
            Box::new(GraphOutputA).into(),
            Some(cn.clone()),
        ));
        (an, cn)
    }
}
