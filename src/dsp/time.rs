use std::sync::Arc;

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
        &mut self,
        _buffer_idx: usize,
        _sample_rate: crate::Scalar,
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
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&std::sync::Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
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
        buf: Vec<Scalar>,
        read_head: Scalar,
        write_head: Scalar,
        delay_current: Scalar,
    }
    @in { input }
    @out { out }
    #in { delay }
    #out {}
}

impl Processor<AudioRate> for Delay {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<AudioRate as super::SignalRate>::SiblingNode>>,
        inputs: &[Signal<AudioRate>],
        outputs: &mut [Signal<AudioRate>],
    ) {
        // kinda a port of:
        // https://github.com/qbroquetas/IV-XDelay/blob/master/IvxDelay/Source/DelayProcessor.cpp

        let cn = sibling_node.as_ref().unwrap();
        let delay_desired_secs = cn.cached_input(0).unwrap().value();

        // self.delay_current =
        //     self.delay_current + 0.00001 * (delay_desired_secs - self.delay_current);
        self.delay_current = delay_desired_secs;

        let sample_offset = self.delay_current * sample_rate;

        self.buf[self.write_head as usize] = inputs[0].value();

        // interpolate
        outputs[0] = Signal::new({
            let mut trunc_read = (self.read_head as usize).min(self.buf.len() - 1);
            let sample0 = self.buf[trunc_read];
            let weight_sample1 = self.read_head - (trunc_read as Scalar);

            trunc_read += 1;
            if trunc_read >= self.buf.len() {
                trunc_read = 0;
            }
            let sample1 = self.buf[trunc_read];
            sample0 + weight_sample1 * (sample1 - sample0)
        });

        self.write_head += 1.0;
        if self.write_head >= self.buf.len() as Scalar {
            self.write_head = 0.0;
        }
        self.read_head = self.write_head - sample_offset;
        if self.read_head < 0.0 {
            self.read_head += self.buf.len() as Scalar;
        }
    }
}

impl Processor<ControlRate> for Delay {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
        _inputs: &[Signal<ControlRate>],
        _outputs: &mut [Signal<ControlRate>],
    ) {
    }
}
