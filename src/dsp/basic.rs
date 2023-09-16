use std::{
    ops::{BitAnd, BitOr, BitXor},
    sync::{Arc, RwLock},
};

use eframe::egui::{Button, Checkbox, Slider, Ui};
use miette::Result;
use papr_proc_macro::node;

use crate::{
    dsp::Signal,
    graph::{Input, Node, NodeName, Output, ProcessorType},
    Scalar,
};

use super::{DspError, Processor, SignalRate};

pub struct Dummy;

impl Processor for Dummy {}

pub struct DebugNode {
    name: String,
}

impl DebugNode {
    pub fn create_node(name: &str) -> Arc<Node> {
        let this = Box::new(RwLock::new(Self {
            name: name.to_string(),
        }));

        Arc::new(Node::new(
            NodeName::new(name),
            vec![Input::new(
                "input".to_owned().as_ref(),
                Some(Signal::new_scalar(0.0)),
            )],
            vec![Output {
                name: "out".to_owned(),
            }],
            crate::graph::ProcessorType::Builtin(this),
        ))
    }
}

impl Processor for DebugNode {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) -> Result<()> {
        let t = inputs[1].expect_scalar()?;
        println!("{} = {:?} (t={t})", self.name, &inputs[0]);
        outputs[0] = inputs[0].clone();
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UiInputWidget {
    Slider { minimum: Scalar, maximum: Scalar },
    Button,
    Toggle,
}

pub struct UiInput {
    pub name: String,
    pub value: Scalar,
    pub widget: UiInputWidget,
}

impl Processor for UiInput {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        _inputs: &[Signal],
        outputs: &mut [Signal],
    ) -> Result<()> {
        outputs[0] = Signal::new_scalar(self.value);
        Ok(())
    }

    fn ui_update(&mut self, ui: &mut Ui) {
        let mut val = self.value;
        match self.widget {
            UiInputWidget::Slider { minimum, maximum } => {
                ui.add(
                    Slider::new(&mut val, minimum..=maximum)
                        .text(&self.name)
                        .step_by(0.0001),
                );
            }
            UiInputWidget::Button => {
                if ui.add(Button::new(&self.name)).clicked() {
                    val = 1.0;
                } else {
                    val = 0.0;
                }
            }
            UiInputWidget::Toggle => {
                let mut b = val > 0.0;
                ui.add(Checkbox::new(&mut b, &self.name));
                if b {
                    val = 1.0;
                } else {
                    val = 0.0;
                }
            }
        }

        self.value = val;
    }
}

impl UiInput {
    pub fn create_node(name: &str, default: Scalar, widget: UiInputWidget) -> Arc<Node> {
        let this = Box::new(RwLock::new(UiInput {
            name: name.to_string(),
            value: default,
            widget,
        }));

        Arc::new(Node::new(
            name.to_string().into(),
            vec![],
            vec![Output {
                name: name.to_string(),
            }],
            ProcessorType::Builtin(this),
        ))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UiOutputWidget {
    Led { value: Scalar },
}

pub struct UiOutput {
    pub name: String,
    pub widget: UiOutputWidget,
}

impl Processor for UiOutput {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) -> Result<()> {
        match &mut self.widget {
            UiOutputWidget::Led { value } => {
                *value = inputs[0].expect_scalar()?;
                outputs[0] = Signal::new_scalar(*value);
            }
        }
        Ok(())
    }

    fn ui_update(&mut self, ui: &mut Ui) {
        match self.widget {
            UiOutputWidget::Led { value } => {
                let _ = ui.radio(value > 0.0, &self.name);
            }
        }
    }
}

impl UiOutput {
    pub fn create_node(name: &str, widget: UiOutputWidget) -> Arc<Node> {
        let this = Box::new(RwLock::new(UiOutput {
            name: name.to_string(),
            widget,
        }));

        let (inputs, outputs) = match widget {
            UiOutputWidget::Led { value } => (
                vec![Input::new("input", Some(Signal::new_scalar(value)))],
                vec![Output {
                    name: "out".to_owned(),
                }],
            ),
        };

        Arc::new(Node::new(
            name.to_string().into(),
            inputs,
            outputs,
            ProcessorType::Builtin(this),
        ))
    }
}

pub struct Constant {
    pub value: Scalar,
}

impl Constant {
    pub fn create_node(name: &str, value: Scalar) -> Arc<Node> {
        let this = Box::new(RwLock::new(Self { value }));

        Arc::new(Node::new(
            NodeName::new(name),
            vec![],
            vec![Output {
                name: "out".to_owned(),
            }],
            crate::graph::ProcessorType::Builtin(this),
        ))
    }
}

impl Processor for Constant {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        _inputs: &[Signal],
        outputs: &mut [Signal],
    ) -> Result<()> {
        outputs[0] = Signal::new_scalar(self.value);
        Ok(())
    }
}

macro_rules! impl_arith {
    ($typ:ident, $op:ident, $use:ident) => {
        node! {
            pub struct $typ;
            in { a, b }
            out { out }
        }

        impl Processor for $typ {
            fn process_sample(
                &mut self,
                _buffer_idx: usize,
                _signal_rate: SignalRate,
                inputs: &[Signal],
                outputs: &mut [Signal],
            ) -> Result<()> {
                use std::ops::$use;
                outputs[0] =
                    Signal::new_scalar(inputs[0].expect_scalar()?.$op(inputs[1].expect_scalar()?));
                Ok(())
            }
        }
    };
}

impl_arith!(Mul, mul, Mul);
impl_arith!(Div, div, Div);
impl_arith!(Add, add, Add);
impl_arith!(Sub, sub, Sub);
impl_arith!(Rem, rem, Rem);

node! {
    pub struct Max;
    in { a, b }
    out { out }

    ~ {
        out = a.max(*b);
    }
}

node! {
    pub struct Min;
    in { a, b }
    out { out }

    ~ {
        out = a.min(*b);
    }
}

node! {
    pub struct Abs;
    in { input }
    out { out }

    ~ {
        out = input.abs();
    }
}

node! {
    pub struct Exp;
    in { input }
    out { out }

    ~ {
        out = input.exp();
    }
}

node! {
    pub struct Cosine;
    in { input }
    out { out }

    ~ {
        out = input.cos();
    }
}

node! {
    pub struct Tanh;
    in { input }
    out { out }

    ~ {
        out = input.tanh();
    }
}

node! {
    pub struct Sine;
    in { input }
    out { out }

    ~ {
        out = input.sin();
    }
}

struct ControlToAudioTx {
    tx: Option<tokio::sync::watch::Sender<Scalar>>,
}
struct ControlToAudioRx {
    rx: Option<tokio::sync::watch::Receiver<Scalar>>,
    value: Scalar,
}

pub struct ControlToAudio;

impl ControlToAudio {
    pub fn create_nodes(name: &str) -> (Arc<Node>, Arc<Node>) {
        let (tx, rx) = tokio::sync::watch::channel(0.0);
        let cn = Arc::new(Node::new(
            name.into(),
            vec![Input::new("c", Some(Signal::new_scalar(0.0)))],
            vec![],
            ProcessorType::Builtin(Box::new(RwLock::new(ControlToAudioTx { tx: Some(tx) }))),
        ));
        let an = Arc::new(Node::new(
            name.into(),
            vec![],
            vec![Output {
                name: "a".to_owned(),
            }],
            ProcessorType::Builtin(Box::new(RwLock::new(ControlToAudioRx {
                rx: Some(rx),
                value: 0.0,
            }))),
        ));
        (an, cn)
    }
}

impl Processor for ControlToAudioTx {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        _outputs: &mut [Signal],
    ) -> Result<()> {
        self.tx
            .as_ref()
            .ok_or(DspError::ChannelDisconnected)?
            .send_replace(inputs[0].expect_scalar()?);
        Ok(())
    }
}
impl Processor for ControlToAudioRx {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        _inputs: &[Signal],
        outputs: &mut [Signal],
    ) -> Result<()> {
        self.value = *self
            .rx
            .as_ref()
            .ok_or(DspError::ChannelDisconnected)?
            .borrow();
        outputs[0] = Signal::new_scalar(self.value);
        Ok(())
    }
}

struct AudioToControlTx {
    tx: Option<tokio::sync::watch::Sender<Scalar>>,
}
struct AudioToControlRx {
    rx: Option<tokio::sync::watch::Receiver<Scalar>>,
    value: Scalar,
}

pub struct AudioToControl;

impl AudioToControl {
    pub fn create_nodes(name: &str) -> (Arc<Node>, Arc<Node>) {
        let (tx, rx) = tokio::sync::watch::channel(0.0);
        let cn = Arc::new(Node::new(
            name.into(),
            vec![],
            vec![Output {
                name: "c".to_owned(),
            }],
            ProcessorType::Builtin(Box::new(RwLock::new(AudioToControlRx {
                rx: Some(rx),
                value: 0.0,
            }))),
        ));
        let an = Arc::new(Node::new(
            name.into(),
            vec![Input::new("a", Some(Signal::new_scalar(0.0)))],
            vec![],
            ProcessorType::Builtin(Box::new(RwLock::new(AudioToControlTx { tx: Some(tx) }))),
        ));
        (an, cn)
    }
}

impl Processor for AudioToControlTx {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        _outputs: &mut [Signal],
    ) -> Result<()> {
        self.tx
            .as_ref()
            .ok_or(DspError::ChannelDisconnected)?
            .send_replace(inputs[0].expect_scalar()?);
        Ok(())
    }
}
impl Processor for AudioToControlRx {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        _inputs: &[Signal],
        outputs: &mut [Signal],
    ) -> Result<()> {
        self.value = *self
            .rx
            .as_ref()
            .ok_or(DspError::ChannelDisconnected)?
            .borrow();
        outputs[0] = Signal::new_scalar(self.value);
        Ok(())
    }
}

node! {
    pub struct RisingEdge {
        c_last: Scalar,
    }
    in { trigger }
    out { out }

    ~ {
        if *trigger > self.c_last {
            out = 1.0;
        } else {
            out = 0.0;
        }
        self.c_last = *trigger;
    }
}

node! {
    pub struct FallingEdge {
        c_last: Scalar,
    }
    in { trigger }
    out { out }

    ~ {
        if *trigger < self.c_last {
            out = 1.0;
        } else {
            out = 0.0;
        }
        self.c_last = *trigger;
    }
}

node! {
    pub struct Clip;
    in { input, low, high }
    out { out }

    ~ {
        out = input.max(*low).min(*high);
    }
}

node! {
    pub struct If;
    in { cmp, then, els }
    out { out }

    ~ {
        if *cmp > 0.0 {
            out = *then;
        } else {
            out = *els;
        }
    }
}

macro_rules! impl_cmp {
    ($id:ident, $op:ident) => {
        node! {
            pub struct $id;
            in { a, b }
            out { out }
        }

        impl Processor for $id {
            fn process_sample(
                &mut self,
                _buffer_idx: usize,
                _signal_rate: SignalRate,
                inputs: &[Signal],
                outputs: &mut [Signal],
            ) -> Result<()> {
                if inputs[0].$op(&inputs[1]) {
                    outputs[0] = Signal::new_scalar(1.0);
                } else {
                    outputs[0] = Signal::new_scalar(0.0);
                }
                Ok(())
            }
        }
    };
}

impl_cmp!(Gt, gt);
impl_cmp!(Lt, lt);
impl_cmp!(Eq, eq);
impl_cmp!(Neq, ne);

macro_rules! impl_boolean {
    ($id:ident, $op:ident) => {
        node! {
            pub struct $id;
            in { a, b }
            out { out }
        }

        impl Processor for $id {
            fn process_sample(
                &mut self,
                _buffer_idx: usize,
                _signal_rate: SignalRate,
                inputs: &[Signal],
                outputs: &mut [Signal],
            ) -> Result<()> {
                if (inputs[0].expect_scalar()? > 0.0).$op(inputs[1].expect_scalar()? > 0.0) {
                    outputs[0] = Signal::new_scalar(1.0);
                } else {
                    outputs[0] = Signal::new_scalar(0.0);
                }
                Ok(())
            }
        }
    };
}

impl_boolean!(And, bitand);
impl_boolean!(Or, bitor);
impl_boolean!(Xor, bitxor);

node! {
    pub struct Not;
    in { input }
    out { out }

    ~ {
        if *input > 0.0 {
            out = 0.0;
        } else {
            out = 1.0;
        }
    }
}
