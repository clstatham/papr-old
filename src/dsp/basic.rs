use std::sync::Arc;

use crate::{
    dsp::SignalImpl,
    graph::{AudioNode, ControlNode, ControlOutput},
    Scalar,
};

use super::{AudioProcessor, AudioSignal, ControlProcessor, ControlSignal};

pub struct DummyControlNode;
impl ControlProcessor for DummyControlNode {
    fn process_control(&self, _t: Scalar, _inputs: &[ControlSignal], _outputs: &[ControlOutput]) {}
}

pub struct DebugControlNode {
    pub name: &'static str,
}

impl ControlProcessor for DebugControlNode {
    fn process_control(&self, t: Scalar, inputs: &[ControlSignal], outputs: &[ControlOutput]) {
        println!("Debug: {} (t={t})", self.name);

        for (inp, out) in inputs.iter().zip(outputs.iter()) {
            println!("{}", inp.value());
            out.tx.send_replace(*inp);
        }
    }
}

pub struct Dac;

impl AudioProcessor for Dac {
    fn into_node(self) -> AudioNode {
        unimplemented!("Use AudioGraph::add_dac() instead of Dac.into_node()")
    }

    fn process_audio(
        &mut self,
        _t: Scalar,
        inputs: &[AudioSignal],
        _control_node: &Arc<ControlNode>,
        outputs: &mut [AudioSignal],
    ) {
        outputs.copy_from_slice(inputs);
    }
}
