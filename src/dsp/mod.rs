use derive_more::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use rustc_hash::FxHashMap;
use std::sync::Arc;

use eframe::egui::Ui;

use crate::{
    graph::{ControlNode, ControlOutput},
    Scalar, PI,
};

pub mod basic;
pub mod generators;

pub trait SignalImpl {
    fn value(self) -> Scalar;
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    PartialOrd,
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Mul,
    MulAssign,
    Div,
    DivAssign,
)]
pub struct AudioSignal(pub Scalar);
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    PartialOrd,
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Mul,
    MulAssign,
    Div,
    DivAssign,
)]
pub struct ControlSignal(pub Scalar);

impl SignalImpl for AudioSignal {
    fn value(self) -> Scalar {
        self.0
    }
}

impl SignalImpl for ControlSignal {
    fn value(self) -> Scalar {
        self.0
    }
}

impl From<Scalar> for AudioSignal {
    fn from(value: Scalar) -> Self {
        Self(value)
    }
}

impl From<Scalar> for ControlSignal {
    fn from(value: Scalar) -> Self {
        Self(value)
    }
}

pub trait AudioProcessor
where
    Self: Send + Sync,
{
    fn process_audio(
        &mut self,
        t: Scalar,
        sample_rate: Scalar,
        inputs: &FxHashMap<String, AudioSignal>,
        control_node: &Arc<ControlNode>,
        outputs: &mut FxHashMap<String, AudioSignal>,
    );
}

pub trait ControlProcessor
where
    Self: Send + Sync,
{
    fn process_control(
        &self,
        t: Scalar,
        control_rate: Scalar,
        inputs: &FxHashMap<String, ControlSignal>,
        outputs: &FxHashMap<String, ControlOutput>,
    );

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
    pub fn new(initial_value: ControlSignal, filter_time_samples: usize) -> Self {
        let cosf = 2.0 - Scalar::cos(2.0 * PI * (2.0 / filter_time_samples as Scalar));
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

    pub fn set_target(&mut self, new_value: ControlSignal) {
        self.target = new_value.value();
        self.xv = self.a0 * new_value.value();
    }

    pub fn next_value(&mut self) -> AudioSignal {
        self.current = self.xv + (self.b1 * self.current);
        AudioSignal(self.current)
    }

    pub fn current_value(&self) -> ControlSignal {
        ControlSignal(self.current)
    }

    pub fn target_value(&self) -> ControlSignal {
        ControlSignal(self.target)
    }
}
