use eframe::egui::Ui;

use crate::{Scalar, TAU};

pub mod basic;
pub mod filters;
pub mod generators;
pub mod graph_util;
pub mod midi;
pub mod time;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignalRate {
    Audio,
    Control,
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Signal {
    val: Scalar,
}

impl std::fmt::Debug for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.val)
    }
}

impl Signal {
    #[inline(always)]
    pub const fn new(val: Scalar) -> Self {
        Self { val }
    }

    #[inline(always)]
    pub const fn value(&self) -> Scalar {
        self.val
    }
}

impl std::ops::Add<Self> for Signal {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.value() + rhs.value())
    }
}

impl std::ops::Sub<Self> for Signal {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.value() - rhs.value())
    }
}

impl std::ops::Mul<Self> for Signal {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.value() * rhs.value())
    }
}

impl std::ops::Div<Self> for Signal {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self::new(self.value() / rhs.value())
    }
}

pub trait Processor {
    #[inline]
    #[allow(unused_variables)]
    fn process_audio_sample(
        &mut self,
        buffer_idx: usize,
        sample_rate: Scalar,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
    }

    #[inline]
    #[allow(unused_variables)]
    fn process_control_sample(
        &mut self,
        buffer_idx: usize,
        sample_rate: Scalar,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
    }

    fn process_buffer(
        &mut self,
        signal_rate: SignalRate,
        sample_rate: Scalar,
        inputs: &[Vec<Signal>],
        outputs: &mut [Vec<Signal>],
    ) {
        let mut audio_buffer_len = inputs
            .iter()
            .next()
            .unwrap_or_else(|| outputs.iter().next().unwrap())
            .len();
        assert!(inputs.iter().all(|inp| {
            let check = inp.len() == audio_buffer_len;
            audio_buffer_len = inp.len();
            check
        }));
        assert!(outputs.iter().all(|out| {
            let check = out.len() == audio_buffer_len;
            audio_buffer_len = out.len();
            check
        }));
        let mut inp = vec![Signal::new(0.0); inputs.len()];
        let mut out = vec![Signal::new(0.0); outputs.len()];
        for i in 0..audio_buffer_len {
            for (val, buf) in inp.iter_mut().zip(inputs) {
                *val = buf[i];
            }
            match signal_rate {
                SignalRate::Audio => self.process_audio_sample(i, sample_rate, &inp, &mut out),
                SignalRate::Control => self.process_control_sample(i, sample_rate, &inp, &mut out),
            };

            for (j, out_val) in out.iter().enumerate() {
                outputs.get_mut(j).unwrap()[i] = *out_val;
            }
        }
    }

    #[allow(unused_variables)]
    fn ui_update(&self, ui: &mut Ui) {}
}

#[non_exhaustive]
pub struct SmoothControlSignal {
    current: Scalar,
    target: Scalar,
    a0: Scalar,
    b1: Scalar,
    xv: Scalar,
}

impl SmoothControlSignal {
    pub fn new(initial_value: Signal, filter_time_samples: usize) -> Self {
        let cosf = 2.0 - Scalar::cos(TAU * (2.0 / filter_time_samples as Scalar));
        let cb1 = cosf - Scalar::sqrt(cosf * cosf - 1.0);
        let mut this = Self {
            current: initial_value.value(),
            target: initial_value.value(),
            a0: 1.0 - cb1,
            b1: cb1,
            xv: 0.0,
        };
        this.set_target(initial_value);
        this
    }

    pub fn set_target(&mut self, new_value: Signal) {
        self.target = new_value.value();
        self.xv = self.a0 * new_value.value();
    }

    pub fn next_value(&mut self) -> Signal {
        self.current = self.xv + (self.b1 * self.current);
        Signal::new(self.current)
    }

    pub fn current_value(&self) -> Signal {
        Signal::new(self.current)
    }

    pub fn target_value(&self) -> Signal {
        Signal::new(self.target)
    }
}
