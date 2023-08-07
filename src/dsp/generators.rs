use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::{
    dsp::{AudioRate, ControlRate},
    graph::{CreateNodes, Input, InputName, Node, Output, OutputName},
    Scalar, PI,
};

use super::{Processor, Signal};

pub struct SineOsc;
impl CreateNodes for SineOsc {
    fn create_nodes() -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let cn = Arc::new(Node::new(
            FxHashMap::from_iter(
                [
                    (InputName("amp".to_owned()), Input::new("amp", 1.0.into())),
                    (
                        InputName("freq".to_owned()),
                        Input::new("freq", 440.0.into()),
                    ),
                ]
                .into_iter(),
            ),
            FxHashMap::default(),
            Box::new(SineOscC).into(),
            None,
        ));
        let an = Arc::new(Node::new(
            FxHashMap::default(),
            FxHashMap::from_iter(
                [(
                    OutputName("out".to_owned()),
                    Output {
                        name: OutputName("out".to_owned()),
                    },
                )]
                .into_iter(),
            ),
            Box::new(SineOscA).into(),
            Some(cn.clone()),
        ));
        (an, cn)
    }
}

pub struct SineOscA;
pub struct SineOscC;

impl Processor<AudioRate> for SineOscA {
    fn process(
        &self,
        t: Scalar,
        _sample_rate: Scalar,
        sibling_node: Option<&Arc<Node<ControlRate>>>,
        _inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        let sibling_node = sibling_node.as_ref().unwrap();
        let amp = sibling_node
            .cached_input(&InputName("amp".to_owned()))
            .unwrap();
        let freq = sibling_node
            .cached_input(&InputName("freq".to_owned()))
            .unwrap();
        *outputs.get_mut(&OutputName("out".to_owned())).unwrap() =
            Signal::new_audio(Scalar::sin(t * PI * 2.0 * freq.value()) * amp.value());
    }
}

impl Processor<ControlRate> for SineOscC {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<AudioRate>>>,
        _inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
    }
}
