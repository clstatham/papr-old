use std::collections::VecDeque;

use papr_proc_macro::node;

use crate::{Scalar, PI};

use super::SignalRate;

const THERMAL: Scalar = 0.000025;

node! {
    pub struct DummyFilter {
        state: VecDeque<Scalar>,
    }
    (inp, cutoff, res) -> (out)

    ~ {
        out = self.state.pop_front().unwrap_or(0.0);
        self.state.push_back(*inp);
    }
}

node! {
    /// Moog ladder lowpass filter, based on https://github.com/ddiakopoulos/MoogLadders/blob/fd147415573e723ba102dfc63dc46af0b7fe55b9/src/HuovilainenModel.h
    pub struct MoogLadder {
        stage: [Scalar; 4],
        stage_tanh: [Scalar; 3],
        delay: [Scalar; 6],
        tune: Scalar,
        acr: Scalar,
        res_quad: Scalar,
    }
    (inp, cutoff, res) -> (out)

    ~ {
        let fc = *cutoff / signal_rate.rate();
        let f = fc * 0.5; // oversample
        let fc2 = fc * fc;
        let fc3 = fc2 * fc;

        let fcr = 1.8730 * fc3 + 0.4955 * fc2 - 0.6490 * fc + 0.9988;
        self.acr = -3.9364 * fc2 + 1.8409 * fc + 0.9968;
        self.tune = (1.0 - Scalar::exp(-((2.0 * PI) * f * fcr))) / THERMAL;
        self.res_quad = 4.0 * *res * self.acr;

        let inp = *inp;

        // oversample
        for _ in 0..2 {
            let mut inp = inp - self.res_quad * self.delay[5];
            self.stage[0] = self.delay[0] + self.tune * (Scalar::tanh(inp * THERMAL) - self.stage_tanh[0]);
            self.delay[0] = self.stage[0];
            for k in 1..4 {
                inp = self.stage[k - 1];
                self.stage_tanh[k - 1] = Scalar::tanh(inp * THERMAL);
                if k == 3 {
                    self.stage[k] = self.delay[k] + self.tune * (self.stage_tanh[k - 1] - Scalar::tanh(self.delay[k] * THERMAL));
                } else {
                    self.stage[k] = self.delay[k] + self.tune * (self.stage_tanh[k - 1] - self.stage_tanh[k]);
                }
                self.delay[k] = self.stage[k];
            }
            self.delay[5] = (self.stage[3] + self.delay[4]) * 0.5;
            self.delay[4] = self.stage[3];
        }

        out = self.delay[5];
    }
}

// node! {
//     /// Moog ladder lowpass filter based on the "linear" implementation from https://github.com/Fredemus/va-filter/blob/4d936956802733732cb86dcacde621e947d0f1a9/src/filter.rs#L125
//     pub struct MoogLadderV1 {
//         vout: [Scalar; 4],
//         state: [Scalar; 4],
//     }
//     (inp, cutoff, res) -> (out)

//     ~ {
//         let cutoff = *cutoff;
//         let res = *res;

//         let g = (PI * cutoff / signal_rate.rate()).tan();
//         // let zeta = 5.0 - 4.9 * res;
//         let k = res.powi(2) * 3.8 - 0.2;

//         let g0 = (1.0 + g).recip();
//         let g1 = g * g0 * g0;
//         let g2 = g * g1 * g0;
//         let g3 = g * g2 * g0;

//         self.vout[3] =
//             (g3 * g * inp + g0 * self.state[3] + g1 * self.state[2] + g2 * self.state[1] + g3 * self.state[0])
//                 / (g3 * g * k + 1.0);
//         self.vout[0] = g0 * (g * (inp - k * self.vout[3]) + self.state[0]);
//         self.vout[1] = g0 * (g * self.vout[0] + self.state[1]);
//         self.vout[2] = g0 * (g * self.vout[1] + self.state[2]);

//         out = self.vout[3];

//         self.update_state();
//     }
// }

// impl MoogLadderV1 {
//     #[inline(always)]
//     fn update_state(&mut self) {
//         self.state[0] = 2.0 * self.vout[0] - self.state[0];
//         self.state[1] = 2.0 * self.vout[1] - self.state[1];
//         self.state[2] = 2.0 * self.vout[2] - self.state[2];
//         self.state[3] = 2.0 * self.vout[3] - self.state[3];
//     }
// }
