use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::{
    dsp::{AudioRate, ControlRate},
    graph::{InputName, Node, OutputName},
    node_constructor, Scalar, PI,
};

use super::{Processor, Signal};

node_constructor! {
    SineOsc
    @in { fm = 0.0 }
    @out { out }
    #in { amp freq fm_amt }
    #out {}
}

impl Processor<AudioRate> for SineOsc {
    fn process(
        &self,
        t: Scalar,
        _sample_rate: Scalar,
        sibling_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        let sibling_node = sibling_node.as_ref().unwrap();
        let amp = sibling_node
            .cached_input(&InputName("amp".to_owned()))
            .unwrap();
        let freq = sibling_node
            .cached_input(&InputName("freq".to_owned()))
            .unwrap();
        let fm_amt = sibling_node
            .cached_input(&InputName("fm_amt".to_owned()))
            .unwrap();
        let fm = inputs[&InputName("fm".to_owned())];
        *outputs.get_mut(&OutputName::default()).unwrap() = Signal::new_audio(
            Scalar::sin(t * PI * 2.0 * freq.value() + fm.value() * PI * 2.0 * fm_amt.value())
                * amp.value(),
        );
    }
}

impl Processor<ControlRate> for SineOsc {
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
