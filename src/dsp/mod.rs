use rustc_hash::FxHashMap;
use std::{marker::PhantomData, sync::Arc};

use eframe::egui::Ui;

use crate::{
    graph::{AudioRate, ControlRate, InputName, OutputName, SignalType},
    Scalar, PI,
};

pub mod basic;
pub mod generators;
pub mod graph_util;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Signal<T: SignalType>(Scalar, PhantomData<T>);

impl<T: SignalType> Signal<T> {
    pub const fn new(val: Scalar) -> Self {
        Self(val, PhantomData)
    }

    pub const fn value(&self) -> Scalar {
        self.0
    }
}

impl Signal<ControlRate> {
    pub const fn new_control(val: Scalar) -> Signal<ControlRate> {
        Self(val, PhantomData)
    }
}

impl Signal<AudioRate> {
    pub const fn new_audio(val: Scalar) -> Signal<AudioRate> {
        Self(val, PhantomData)
    }
}

impl<T: SignalType> From<Scalar> for Signal<T> {
    fn from(value: Scalar) -> Self {
        Self::new(value)
    }
}

impl<T: SignalType> std::ops::Add<Self> for Signal<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.value() + rhs.value())
    }
}

impl<T: SignalType> std::ops::Sub<Self> for Signal<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.value() - rhs.value())
    }
}

impl<T: SignalType> std::ops::Mul<Self> for Signal<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.value() * rhs.value())
    }
}

impl<T: SignalType> std::ops::Div<Self> for Signal<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self::new(self.value() / rhs.value())
    }
}

pub trait Processor<T: SignalType>
where
    Self: Send + Sync,
{
    fn process(
        &self,
        t: Scalar,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<T::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<T>>,
        outputs: &mut FxHashMap<OutputName, Signal<T>>,
    );

    fn ui_update(&self, _ui: &mut Ui) {}
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
    pub fn new(initial_value: Signal<ControlRate>, filter_time_samples: usize) -> Self {
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

    pub fn set_target(&mut self, new_value: Signal<ControlRate>) {
        self.target = new_value.value();
        self.xv = self.a0 * new_value.value();
    }

    pub fn next_value(&mut self) -> Signal<AudioRate> {
        self.current = self.xv + (self.b1 * self.current);
        Signal::new_audio(self.current)
    }

    pub fn current_value(&self) -> Signal<ControlRate> {
        Signal::new_control(self.current)
    }

    pub fn target_value(&self) -> Signal<ControlRate> {
        Signal::new_control(self.target)
    }
}
