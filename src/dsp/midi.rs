use papr_proc_macro::node_constructor;

use crate::Scalar;

use super::{Processor, Signal};

node_constructor! {
    pub struct MidiToFreq;
    in { m }
    out { f }
}

impl Processor for MidiToFreq {
    fn process_control_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,

        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        let midi = inputs[0];
        *outputs.get_mut(0).unwrap() =
            Signal::new((2.0 as Scalar).powf((midi.value() - 69.0) / 12.0) * 440.0);
    }
}
