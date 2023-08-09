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
    #in { m }
    #out { f }
}

impl Processor<AudioRate> for MidiToFreq {
    fn process(
        &self,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as SignalType>::SiblingNode>>,
        _inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
    }
}

impl Processor<ControlRate> for MidiToFreq {
    fn process(
        &self,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        let midi = inputs[&InputName("m".to_owned())];
        *outputs.get_mut(&OutputName("f".to_owned())).unwrap() =
            ((2.0 as Scalar).powf((midi.value() - 69.0) / 12.0) * 440.0).into();
    }
}
