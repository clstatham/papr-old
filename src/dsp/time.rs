use papr_proc_macro::node_constructor;

use crate::Scalar;

use super::{Processor, Signal, SignalRate};

node_constructor! {
    pub struct Clock;
    in { period, width }
    out { out }
}

impl Processor for Clock {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[super::Signal],
        outputs: &mut [super::Signal],
    ) {
        let t = inputs[Self::input_idx("t").unwrap()];
        let period = inputs[0];
        let width = inputs[1];
        if period.value() == 0.0 {
            outputs[0] = Signal::new(0.0);
        } else if t.value() % period.value() < period.value() * width.value() {
            outputs[0] = Signal::new(1.0);
        } else {
            outputs[0] = Signal::new(0.0);
        }
    }
}

node_constructor! {
    pub struct Delay {
        buf: Vec<Scalar>,
        read_head: Scalar,
        write_head: Scalar,
        delay_current: Scalar,
    }
    in { input, delay }
    out { out }
}

impl Processor for Delay {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        // kinda a port of:
        // https://github.com/qbroquetas/IV-XDelay/blob/master/IvxDelay/Source/DelayProcessor.cpp

        let delay_desired_secs = inputs[1].value();

        self.delay_current =
            self.delay_current + 0.00005 * (delay_desired_secs - self.delay_current);
        // self.delay_current = delay_desired_secs;

        let sample_offset = self.delay_current * signal_rate.rate();

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
