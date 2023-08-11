use std::sync::{Arc, Mutex};

use papr_proc_macro::node_constructor;

use crate::Scalar;

use super::{AudioRate, ControlRate, Processor, Signal};

node_constructor! {
    pub struct Clock;
    @in {}
    @out {}
    #in { period, width }
    #out { out }
}

impl Processor<AudioRate> for Clock {
    fn process_sample(
        &self,
        buffer_idx: usize,
        sample_rate: crate::Scalar,
        sibling_node: Option<&std::sync::Arc<<AudioRate as super::SignalRate>::SiblingNode>>,
        inputs: &[super::Signal<AudioRate>],
        outputs: &mut [super::Signal<AudioRate>],
    ) {
    }
}

impl Processor<ControlRate> for Clock {
    fn process_sample(
        &self,
        buffer_idx: usize,
        sample_rate: Scalar,
        sibling_node: Option<&std::sync::Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
        inputs: &[super::Signal<ControlRate>],
        outputs: &mut [super::Signal<ControlRate>],
    ) {
        let t = inputs[Self::control_input_idx("t").unwrap()];
        let period = inputs[0];
        let width = inputs[1];
        if t.value() % period.value() < period.value() * width.value() {
            outputs[0] = Signal::new(1.0);
        } else {
            outputs[0] = Signal::new(0.0);
        }
        dbg!(outputs[0]);
    }
}
