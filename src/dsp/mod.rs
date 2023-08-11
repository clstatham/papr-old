use std::{marker::PhantomData, sync::Arc};

use eframe::egui::Ui;

use crate::{graph::Node, Scalar, TAU};

pub mod basic;
pub mod filters;
pub mod generators;
pub mod graph_util;
pub mod midi;
pub mod time;

pub trait SignalRate
where
    Self: Copy,
{
    type SiblingNode;
}

#[derive(Clone, Copy, Debug)]
pub struct AudioRate;
#[derive(Clone, Copy, Debug)]
pub struct ControlRate;
impl SignalRate for AudioRate {
    type SiblingNode = Node<ControlRate>;
}
impl SignalRate for ControlRate {
    type SiblingNode = Node<AudioRate>;
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Signal<T: SignalRate>(Scalar, PhantomData<T>);

impl std::fmt::Debug for Signal<AudioRate> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "@{}", self.0)
    }
}

impl std::fmt::Debug for Signal<ControlRate> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

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
        inputs: &[Signal<T>],
        outputs: &mut [Signal<T>],
    );

    fn process_buffer(
        &self,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<T::SiblingNode>>,
        inputs: &[Vec<Signal<T>>],
        outputs: &mut [Vec<Signal<T>>],
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
        // let mut inp = Vec::from_iter(inputs.iter().map(|(name, _inp)| (*name, Signal::new(0.0))));
        // let mut out = Vec::from_iter(outputs.iter().map(|(name, _out)| (*name, Signal::new(0.0))));
        let mut inp = vec![Signal::new(0.0); inputs.len()];
        let mut out = vec![Signal::new(0.0); outputs.len()];
        for i in 0..audio_buffer_len {
            for (val, buf) in inp.iter_mut().zip(inputs) {
                *val = buf[i];
            }
            self.process_sample(i, sample_rate, sibling_node, &inp, &mut out);
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
