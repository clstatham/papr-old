use miette::Result;
use papr_proc_macro::node_constructor;

use crate::Scalar;

use super::{Processor, Signal, SignalRate};

node_constructor! {
    pub struct MidiToFreq;
    in { m }
    out { f }
}

impl Processor for MidiToFreq {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) -> Result<()> {
        let midi = inputs[0];
        outputs[0] = Signal::new((2.0 as Scalar).powf((midi.value() - 69.0) / 12.0) * 440.0);
        Ok(())
    }
}
