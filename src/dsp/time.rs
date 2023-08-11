use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use papr_proc_macro::node_constructor;

use crate::Scalar;

use super::{AudioRate, ControlRate, Processor, Signal};

node_constructor! {
    pub struct Clock;
    @in {}
    @out { out }
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
        let cn = sibling_node.as_ref().unwrap();
        let t = inputs[Self::audio_input_idx("t").unwrap()];
        let period = cn.cached_input(0).unwrap();
        let width = cn.cached_input(1).unwrap();
        if period.value() == 0.0 {
            outputs[0] = Signal::new(0.0);
        } else if t.value() % period.value() < period.value() * width.value() {
            outputs[0] = Signal::new(1.0);
        } else {
            outputs[0] = Signal::new(0.0);
        }
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
        // dbg!(outputs[0]);
    }
}

node_constructor! {
    pub struct Delay {
        // write_head: Arc<Mutex<usize>>,
        buf: Arc<Mutex<VecDeque<Scalar>>>,
    }
    @in { input }
    @out { out }
    #in { delay }
    #out {}
}

impl Processor<AudioRate> for Delay {
    fn process_sample(
        &self,
        buffer_idx: usize,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<AudioRate as super::SignalRate>::SiblingNode>>,
        inputs: &[Signal<AudioRate>],
        outputs: &mut [Signal<AudioRate>],
    ) {
        let mut buf = self.buf.lock().unwrap();
        // let mut write_head = self.write_head.lock().unwrap();
        let cn = sibling_node.as_ref().unwrap();
        let delay = cn.cached_input(0).unwrap();
        let delay_samps = (delay.value() * sample_rate) as usize;
        buf.push_back(0.0);
        buf[delay_samps] = inputs[0].value();
        outputs[0] = Signal::new_audio(buf.pop_front().unwrap()); // "read" head
    }
}

impl Processor<ControlRate> for Delay {
    fn process_sample(
        &self,
        buffer_idx: usize,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
        inputs: &[Signal<ControlRate>],
        outputs: &mut [Signal<ControlRate>],
    ) {
    }
}
