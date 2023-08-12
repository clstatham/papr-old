use crate::{node_constructor, Scalar, PI, TAU};

use super::{Processor, Signal};

node_constructor! {
    pub struct SineOsc;
    in { amp, freq, fm_amt, fm }
    out { out }
}

impl Processor for SineOsc {
    fn process_audio_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        let amp = inputs[0];
        let freq = inputs[1];
        let fm_amt = inputs[2];
        let t = inputs[Self::input_idx("t").unwrap()].value();
        let fm = inputs[3];
        *outputs.get_mut(0).unwrap() = Signal::new(
            Scalar::sin(t * TAU * freq.value() + fm.value() * TAU * fm_amt.value()) * amp.value(),
        );
    }
}

node_constructor! {
    pub struct SineOscLFO;
    in { amp, freq }
    out { out }
}

impl Processor for SineOscLFO {
    fn process_control_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        let amp = inputs[0];
        let freq = inputs[1];
        let t = inputs[Self::input_idx("t").unwrap()].value();
        *outputs.get_mut(0).unwrap() =
            Signal::new(Scalar::sin(t * TAU * freq.value()) * amp.value());
    }
}

node_constructor! {
    pub struct BlSawOsc {
        p: Scalar,
        /// SET TO 1.0 INITIALLY
        dp: Scalar,
        saw: Scalar,
    }
    in { amp, freq }
    out { out }
}

impl Processor for BlSawOsc {
    fn process_audio_sample(
        &mut self,
        _buffer_idx: usize,
        sample_rate: Scalar,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        let amp = inputs[0];
        let freq = inputs[1];

        // algorithm courtesy of https://www.musicdsp.org/en/latest/Synthesis/12-bandlimited-waveforms.html

        let pmax = 0.5 * sample_rate / freq.value();
        let dc = -0.498 / pmax;

        self.p += self.dp;
        if self.p < 0.0 {
            self.p = -self.p;
            self.dp = -self.dp;
        } else if self.p > pmax {
            self.p = pmax + pmax - self.p;
            self.dp = -self.dp;
        }
        let mut x = PI * self.p;
        if x < 0.00001 {
            x = 0.00001;
        }
        self.saw = 0.995 * self.saw + dc + x.sin() / x;

        outputs[0] = Signal::new(self.saw * amp.value());
    }
}
