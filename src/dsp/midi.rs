use std::sync::Arc;

use papr_proc_macro::node_constructor;
use rustc_hash::FxHashMap;

use crate::{
    graph::{InputName, OutputName},
    Scalar,
};

use super::{AudioRate, ControlRate, Processor, Signal, SignalType};

node_constructor! {
    pub struct MidiToFreq;
    @in {}
    @out {}
    #in { midi }
    #out { freq }
}

impl Processor<AudioRate> for MidiToFreq {
    fn process(
        &self,
        t: Scalar,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<AudioRate as SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
    }
}

impl Processor<ControlRate> for MidiToFreq {
    fn process(
        &self,
        t: Scalar,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<ControlRate as SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        let midi = inputs[&InputName("midi".to_owned())];
        *outputs.get_mut(&OutputName("freq".to_owned())).unwrap() =
            ((2.0 as Scalar).powf((midi.value() - 69.0) / 12.0) * 440.0).into();
    }
}
