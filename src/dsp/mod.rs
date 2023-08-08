use rustc_hash::FxHashMap;
use std::{marker::PhantomData, sync::Arc};

use eframe::egui::Ui;

use crate::{
    graph::{InputName, Node, OutputName},
    Scalar, PI,
};

pub mod basic;
pub mod generators;
pub mod graph_util;

pub trait SignalType
where
    Self: Copy,
{
    type SiblingNode;
}

#[derive(Clone, Copy)]
pub struct AudioRate;
#[derive(Clone, Copy)]
pub struct ControlRate;
impl SignalType for AudioRate {
    type SiblingNode = Node<ControlRate>;
}
impl SignalType for ControlRate {
    type SiblingNode = Node<AudioRate>;
}

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

// magic macro to make implementing new nodes way easier
#[macro_export]
macro_rules! node_constructor {
    {
        $name:ident
        @in {$($audio_inputs:ident = $ai_default_values:expr)*}
        @out {$($audio_outputs:ident)*}
        #in {$($control_inputs:ident)*}
        #out {$($control_outputs:ident)*}
    } => {
        pub struct $name;
        impl $name {
            pub fn create_nodes(name: &str, $($control_inputs : Scalar),*) -> (std::sync::Arc<$crate::graph::Node<$crate::dsp::AudioRate>>, std::sync::Arc<$crate::graph::Node<$crate::dsp::ControlRate>>) {
                let a_outs = [$((
                    $crate::graph::OutputName(stringify!($audio_outputs).to_owned()),
                    $crate::graph::Output { name: $crate::graph::OutputName(stringify!($audio_outputs).to_owned()) }
                )),*];
                let c_outs = [$((
                    $crate::graph::OutputName(stringify!($control_outputs).to_owned()),
                    $crate::graph::Output { name: $crate::graph::OutputName(stringify!($control_outputs).to_owned()) }
                )),*];
                let a_ins = [$((
                    $crate::graph::InputName(stringify!($audio_inputs).to_owned()),
                    $crate::graph::Input::new(stringify!($audio_inputs), Some($crate::dsp::Signal::new($ai_default_values)))
                )),*];
                let c_ins = [$((
                    $crate::graph::InputName(stringify!($control_inputs).to_owned()),
                    $crate::graph::Input::new(stringify!($control_inputs), Some($crate::dsp::Signal::new_control($control_inputs)))
                )),*];

                let cn = std::sync::Arc::new($crate::graph::Node::new(
                    $crate::graph::NodeName(name.to_owned()),
                    FxHashMap::from_iter(c_ins.into_iter()),
                    FxHashMap::from_iter(c_outs.into_iter()),
                    $crate::graph::ProcessorType::Boxed(Box::new(Self)),
                    None,
                ));
                let an = std::sync::Arc::new($crate::graph::Node::new(
                    $crate::graph::NodeName(name.to_owned()),
                    FxHashMap::from_iter(a_ins.into_iter()),
                    FxHashMap::from_iter(a_outs.into_iter()),
                    $crate::graph::ProcessorType::Boxed(Box::new(Self)),
                    Some(cn.clone()),
                ));
                (an, cn)
            }
        }
    };
}
