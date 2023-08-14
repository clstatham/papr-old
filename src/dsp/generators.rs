use crate::{Scalar, PI, TAU};

use papr_proc_macro::node_constructor;

use super::{Processor, Signal, SignalRate};

node_constructor! {
    pub struct FmSineOsc;
    in { amp, freq, fm_amt, fm }
    out { out }
}

impl Processor for FmSineOsc {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        let amp = inputs[0];
        let freq = inputs[1];
        let fm_amt = inputs[2];
        let t = inputs[Self::input_idx("t").unwrap()].value();
        let fm = inputs[3];

        if freq.value() <= 0.0 {
            return;
        }

        outputs[0] = Signal::new(
            Scalar::sin(t * TAU * freq.value() + fm.value() * TAU * fm_amt.value()) * amp.value(),
        );
    }
}

node_constructor! {
    pub struct SineOsc;
    in { amp, freq }
    out { out }
}

impl Processor for SineOsc {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        let amp = inputs[0];
        let freq = inputs[1];

        if freq.value() <= 0.0 {
            return;
        }

        let t = inputs[Self::input_idx("t").unwrap()].value();
        outputs[0] = Signal::new(Scalar::sin(t * TAU * freq.value()) * amp.value());
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
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        let amp = inputs[0];
        let freq = inputs[1];

        if freq.value() <= 0.0 {
            return;
        }

        // algorithm courtesy of https://www.musicdsp.org/en/latest/Synthesis/12-bandlimited-waveforms.html

        let pmax = 0.5 * signal_rate.rate() / freq.value();
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

pub const BL_SQUARE_MAX_COEFF: usize = 48000 / (5 * 4);

node_constructor! {
    pub struct BlSquareOsc {
        coeff: [Scalar; BL_SQUARE_MAX_COEFF],
    }
    in { amp, freq, d }
    out { out }
}

impl Processor for BlSquareOsc {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        let t = inputs[Self::input_idx("t").unwrap()].value();
        let amp = inputs[0].value();
        let freq = inputs[1].value();
        let d = inputs[2].value();
        let sr = signal_rate.rate();

        if freq <= 0.0 {
            return;
        }

        let n_harm = (sr / (freq * 4.0)) as usize;
        self.coeff[0] = d - 0.5;
        for i in 1..n_harm + 1 {
            self.coeff[i] = Scalar::sin(i as Scalar * d * PI) * 2.0 / (i as Scalar * PI);
        }
        let theta = t * TAU * freq;
        outputs[0] = Signal::new(
            amp * (self
                .coeff
                .iter()
                .take(n_harm + 1)
                .enumerate()
                .map(|(i, coeff)| coeff * Scalar::cos(i as Scalar * theta)))
            .sum::<Scalar>(),
        );
    }
}
