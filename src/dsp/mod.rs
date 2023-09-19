use eframe::egui::Ui;
use miette::{Diagnostic, Result};
use thiserror::Error;

use crate::Scalar;

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
    #[error("Expected scalar signal, got {0:?}")]
    ExpectedScalar(Signal),
    #[error("Expected array signal, got {0:?}")]
    ExpectedArray(Signal),
    #[error("Expected symbol signal, got {0:?}")]
    ExpectedSymbol(Signal),
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalType {
    Scalar,
    Symbol,
    Array,
}

#[derive(Clone, PartialEq, PartialOrd)]
pub enum Signal {
    Scalar(Scalar),
    Symbol(String),
    Array(Vec<Scalar>),
}

impl std::fmt::Debug for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Signal::Scalar(val) => write!(f, "{:?}", val),
            Signal::Symbol(val) => write!(f, "{:?}", val),
            Signal::Array(val) => write!(f, "{:?}", val),
        }
    }
}

impl Signal {
    #[inline(always)]
    pub const fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }

    #[inline(always)]
    pub const fn is_symbol(&self) -> bool {
        matches!(self, Self::Symbol(_))
    }

    #[inline(always)]
    pub const fn is_array(&self) -> bool {
        matches!(self, Self::Array(_))
    }

    #[inline(always)]
    pub const fn scalar_value(&self) -> Option<Scalar> {
        match self {
            Self::Scalar(val) => Some(*val),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn expect_scalar(&self) -> Result<Scalar> {
        match self {
            Self::Scalar(val) => Ok(*val),
            _ => Err(DspError::ExpectedScalar(self.clone()).into()),
        }
    }

    #[inline(always)]
    pub fn symbol_value(&self) -> Option<&str> {
        match self {
            Self::Symbol(val) => Some(val),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn expect_symbol(&self) -> Result<&str> {
        match self {
            Self::Symbol(val) => Ok(val),
            _ => Err(DspError::ExpectedSymbol(self.clone()).into()),
        }
    }

    #[inline(always)]
    pub fn array_value(&self) -> Option<&[Scalar]> {
        match self {
            Self::Array(val) => Some(val),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn expect_array(&self) -> Result<&[Scalar]> {
        match self {
            Self::Array(val) => Ok(val),
            _ => Err(DspError::ExpectedArray(self.clone()).into()),
        }
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
        let mut buffer_len = if let Some(len) = inputs
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
            let check = inp.len() == buffer_len;
            buffer_len = inp.len();
            check
        }));
        assert!(outputs.iter().all(|out| {
            let check = out.len() == buffer_len;
            buffer_len = out.len();
            check
        }));
        let mut inp = vec![Signal::Scalar(0.0); inputs.len()];
        let mut out = vec![Signal::Scalar(0.0); outputs.len()];
        for i in 0..buffer_len {
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
