use rustc_hash::FxHashMap;
use std::{marker::PhantomData, sync::Arc};

use eframe::egui::Ui;

use crate::{
    graph::{InputName, Node, OutputName},
    Scalar, TAU,
};

pub mod basic;
pub mod filters;
pub mod generators;
pub mod graph_util;
pub mod midi;

pub trait SignalRate
where
    Self: Copy,
{
    type SiblingNode;
}

#[derive(Clone, Copy)]
pub struct AudioRate;
#[derive(Clone, Copy)]
pub struct ControlRate;
impl SignalRate for AudioRate {
    type SiblingNode = Node<ControlRate>;
}
impl SignalRate for ControlRate {
    type SiblingNode = Node<AudioRate>;
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Signal<T: SignalRate>(Scalar, PhantomData<T>);

impl<T: SignalRate> Signal<T> {
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

impl<T: SignalRate> From<Scalar> for Signal<T> {
    fn from(value: Scalar) -> Self {
        Self::new(value)
    }
}

impl<T: SignalRate> std::ops::Add<Self> for Signal<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.value() + rhs.value())
    }
}

impl<T: SignalRate> std::ops::Sub<Self> for Signal<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.value() - rhs.value())
    }
}

impl<T: SignalRate> std::ops::Mul<Self> for Signal<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.value() * rhs.value())
    }
}

impl<T: SignalRate> std::ops::Div<Self> for Signal<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self::new(self.value() / rhs.value())
    }
}

pub trait Processor<T: SignalRate>
where
    Self: Send + Sync,
{
    fn process_sample(
        &self,
        buffer_idx: usize,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<T::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<T>>,
        outputs: &mut FxHashMap<OutputName, Signal<T>>,
    );

    fn process_buffer(
        &self,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<T::SiblingNode>>,
        inputs: &FxHashMap<InputName, Vec<Signal<T>>>,
        outputs: &mut FxHashMap<OutputName, Vec<Signal<T>>>,
    ) {
        let mut audio_buffer_len = inputs.iter().next().unwrap().1.len();
        assert!(inputs.iter().all(|(_, inp)| {
            let check = inp.len() == audio_buffer_len;
            audio_buffer_len = inp.len();
            check
        }));
        assert!(outputs.iter().all(|(_, inp)| {
            let check = inp.len() == audio_buffer_len;
            audio_buffer_len = inp.len();
            check
        }));
        let mut inp = FxHashMap::from_iter(
            inputs
                .iter()
                .map(|(name, _inp)| (name.to_owned(), Signal::new(0.0))),
        );
        let mut out = FxHashMap::from_iter(
            outputs
                .iter()
                .map(|(name, _out)| (name.to_owned(), Signal::new(0.0))),
        );
        for i in 0..audio_buffer_len {
            for (name, val) in inp.iter_mut() {
                *val = inputs[name][i];
            }
            self.process_sample(i, sample_rate, sibling_node, &inp, &mut out);
            for (out_name, out_val) in &out {
                outputs.get_mut(out_name).unwrap()[i] = *out_val;
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
    pub fn new(initial_value: Signal<ControlRate>, filter_time_samples: usize) -> Self {
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
