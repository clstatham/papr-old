use std::sync::{Arc, RwLock};

use rustc_hash::FxHashMap;

use crate::{
    graph::{AudioRate, ControlRate, CreateNodes, Input, InputName, Node, Output, OutputName},
    Scalar,
};

use super::{basic::DummyC, Processor, Signal};

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

impl CreateNodes for GraphInput {
    fn create_nodes() -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let cn = Arc::new(Node::new(
            FxHashMap::default(),
            FxHashMap::default(),
            Box::new(DummyC),
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
            Box::new(GraphInputA),
            Some(cn.clone()),
        ));
        (an, cn)
    }
}
