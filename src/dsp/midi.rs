use std::sync::Arc;

use papr_proc_macro::node_constructor;


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
        _inputs: &[Signal<AudioRate>],
        _outputs: &mut [Signal<AudioRate>],
    ) {
    }
}

impl Processor<ControlRate> for MidiToFreq {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as SignalRate>::SiblingNode>>,
        inputs: &[Signal<ControlRate>],
        outputs: &mut [Signal<ControlRate>],
    ) {
        let midi = inputs[0];
        *outputs.get_mut(0).unwrap() =
            ((2.0 as Scalar).powf((midi.value() - 69.0) / 12.0) * 440.0).into();
    }
}
