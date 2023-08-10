use std::sync::Arc;

use papr_proc_macro::node_constructor;
use std::collections::BTreeMap;

use crate::Scalar;

use super::{AudioRate, ControlRate, Processor, Signal, SignalRate};

node_constructor! {
    pub struct MidiToFreq;
    @in {}
    @out {}
    #in { m }
    #out { f }
}

impl Processor<AudioRate> for MidiToFreq {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as SignalRate>::SiblingNode>>,
        _inputs: &BTreeMap<&str, Signal<AudioRate>>,
        _outputs: &mut BTreeMap<&str, Signal<AudioRate>>,
    ) {
    }
}

impl Processor<ControlRate> for MidiToFreq {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as SignalRate>::SiblingNode>>,
        inputs: &BTreeMap<&str, Signal<ControlRate>>,
        outputs: &mut BTreeMap<&str, Signal<ControlRate>>,
    ) {
        let midi = inputs["m"];
        *outputs.get_mut("f").unwrap() =
            ((2.0 as Scalar).powf((midi.value() - 69.0) / 12.0) * 440.0).into();
    }
}
