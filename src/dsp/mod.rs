use eframe::egui::Ui;
use miette::{Diagnostic, Result};
use thiserror::Error;

use crate::{Scalar, TAU};

pub mod basic;
pub mod filters;
pub mod generators;
pub mod graph_util;
pub mod midi;
pub mod samplers;
pub mod time;

#[derive(Debug, Error, Diagnostic)]
pub enum DspError {
    #[error("Decoder error: {0}")]
    Creak(#[from] creak::DecoderError),
    #[error("Processing error: {0}")]
    Processing(String),
    #[error("Channel disconnected")]
    ChannelDisconnected,
    #[error("No input named {0}")]
    NoInputNamed(String),
    #[error("No output named {0}")]
    NoOutputNamed(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalRate {
    Audio {
        sample_rate: Scalar,
        buffer_len: usize,
    },
    Control {
        sample_rate: Scalar,
        buffer_len: usize,
    },
}

impl SignalRate {
    #[inline(always)]
    pub fn rate(self) -> Scalar {
        match self {
            Self::Audio { sample_rate, .. } => sample_rate,
            Self::Control { sample_rate, .. } => sample_rate,
        }
    }

    #[inline(always)]
    pub fn buffer_len(self) -> usize {
        match self {
            Self::Audio { buffer_len, .. } => buffer_len,
            Self::Control { buffer_len, .. } => buffer_len,
        }
    }
}

#[derive(Clone, PartialEq, PartialOrd)]
pub enum Signal {
    Scalar(Scalar),
    Symbol(String),
}

impl std::fmt::Debug for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Signal::Scalar(val) => write!(f, "{:?}", val),
            Signal::Symbol(val) => write!(f, "{:?}", val),
        }
    }
}

impl Signal {
    #[inline(always)]
    pub const fn new_scalar(val: Scalar) -> Self {
        Self::Scalar(val)
    }

    #[inline(always)]
    pub const fn scalar_value(&self) -> Scalar {
        match self {
            Self::Scalar(val) => *val,
            Self::Symbol(_) => 0.0,
        }
    }

    #[inline(always)]
    pub fn new_symbol(val: &str) -> Self {
        Self::Symbol(val.to_string())
    }

    #[inline(always)]
    pub fn symbol_value(&self) -> &str {
        match self {
            Self::Scalar(_) => "",
            Self::Symbol(val) => val,
        }
    }
}

impl std::ops::Add<Self> for Signal {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new_scalar(self.scalar_value() + rhs.scalar_value())
    }
}

impl std::ops::Sub<Self> for Signal {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new_scalar(self.scalar_value() - rhs.scalar_value())
    }
}

impl std::ops::Mul<Self> for Signal {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new_scalar(self.scalar_value() * rhs.scalar_value())
    }
}

impl std::ops::Div<Self> for Signal {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self::new_scalar(self.scalar_value() / rhs.scalar_value())
    }
}

pub trait Processor {
    #[inline]
    #[allow(unused_variables)]
    fn process_sample(
        &mut self,
        buffer_idx: usize,
        signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) -> Result<()> {
        Ok(())
    }

    fn process_buffer(
        &mut self,
        signal_rate: SignalRate,
        inputs: &[Vec<Signal>],
        outputs: &mut [Vec<Signal>],
    ) -> Result<()> {
        let mut audio_buffer_len = if let Some(len) = inputs
            .iter()
            .next()
            .or_else(|| outputs.iter().next())
            .map(|x| x.len())
        {
            len
        } else {
            // no inputs/outputs
            return Ok(());
        };
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
        let mut inp = vec![Signal::new_scalar(0.0); inputs.len()];
        let mut out = vec![Signal::new_scalar(0.0); outputs.len()];
        for i in 0..audio_buffer_len {
            for (val, buf) in inp.iter_mut().zip(inputs) {
                *val = buf[i].clone();
            }

            self.process_sample(i, signal_rate, &inp, &mut out)?;

            for (j, out_val) in out.iter().enumerate() {
                outputs
                    .get_mut(j)
                    .ok_or(DspError::NoOutputNamed(j.to_string()))?[i] = out_val.clone();
            }
        }
        Ok(())
    }

    #[allow(unused_variables)]
    fn ui_update(&mut self, ui: &mut Ui) {}
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
            current: initial_value.scalar_value(),
            target: initial_value.scalar_value(),
            a0: 1.0 - cb1,
            b1: cb1,
            xv: 0.0,
        };
        this.set_target(initial_value);
        this
    }

    pub fn set_target(&mut self, new_value: Signal) {
        self.target = new_value.scalar_value();
        self.xv = self.a0 * new_value.scalar_value();
    }

    pub fn next_value(&mut self) -> Signal {
        self.current = self.xv + (self.b1 * self.current);
        Signal::new_scalar(self.current)
    }

    pub fn current_value(&self) -> Signal {
        Signal::new_scalar(self.current)
    }

    pub fn target_value(&self) -> Signal {
        Signal::new_scalar(self.target)
    }
}
