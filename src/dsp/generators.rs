use crate::{Scalar, PI, TAU};

use miette::Result;
use papr_proc_macro::node;

use super::{DspError, Processor, Signal, SignalRate};

node! {
    pub struct FmSineOsc;
    (amp, freq, fm_amt, fm) -> (out)

    ~ {
        if *freq < 0.0 {
            return Ok(());
        }

        out = amp * Scalar::sin(t * TAU * freq + fm * TAU * fm_amt);
    }
}

node! {
    pub struct SineOsc;
    (amp, freq) -> (out)

    ~ {
        if *freq < 0.0 {
            return Ok(());
        }

        out = amp * Scalar::sin(t * TAU * freq);
    }
}

node! {
    pub struct BlSawOsc {
        p: Scalar,
        /// SET TO 1.0 INITIALLY
        dp: Scalar,
        saw: Scalar,
    }
    (amp, freq) -> (out)

    ~ {
        if *freq <= 0.0 {
            return Ok(());
        }

        // algorithm courtesy of https://www.musicdsp.org/en/latest/Synthesis/12-bandlimited-waveforms.html

        let pmax = 0.5 * signal_rate.rate() / *freq;
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

        out = self.saw * amp;
    }
}

pub const BL_SQUARE_MAX_COEFF: usize = 48000 / (5 * 4);

node! {
    pub struct BlSquareOsc {
        coeff: [Scalar; BL_SQUARE_MAX_COEFF],
    }
    (amp, freq, d) -> (out)

    ~ {
        if *freq <= 0.0 {
            return Ok(());
        }

        let n_harm = (signal_rate.rate() / (freq * 4.0)) as usize;
        self.coeff[0] = *d - 0.5;
        for i in 1..n_harm + 1 {
            self.coeff[i] = Scalar::sin(i as Scalar * *d * PI) * 2.0 / (i as Scalar * PI);
        }
        let theta = t * TAU * freq;
        out = amp * (self
            .coeff
            .iter()
            .take(n_harm + 1)
            .enumerate()
            .map(|(i, coeff)| coeff * Scalar::cos(i as Scalar * theta)))
        .sum::<Scalar>();
    }
}
