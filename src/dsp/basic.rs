use std::{
    ops::{BitAnd, BitOr, BitXor},
    sync::{Arc, RwLock},
};

use eframe::egui::{Slider, Ui};
use papr_proc_macro::node_constructor;

use crate::{
    dsp::Signal,
    graph::{Input, Node, NodeName, Output, ProcessorType},
    Scalar,
};

use super::{Processor, SignalRate};

pub struct Dummy;

impl Processor for Dummy {}

pub struct DebugNode {
    name: String,
}

impl DebugNode {
    pub fn create_node(name: &str, buffer_len: usize) -> Arc<Node> {
        let this = Box::new(RwLock::new(Self {
            name: name.to_string(),
        }));

        Arc::new(Node::new(
            NodeName::new(name),
            buffer_len,
            vec![Input::new(
                "input".to_owned().as_ref(),
                Some(Signal::new(0.0)),
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
    ) {
        let t = inputs[1].value();
        println!("{} = {} (t={t})", self.name, inputs[0].value());
        outputs[0] = inputs[0];
    }
}

pub struct UiInput {
    pub name: String,
    pub minimum: Signal,
    pub maximum: Signal,
    pub value: Signal,
}

impl Processor for UiInput {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        _inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = self.value;
    }

    fn ui_update(&mut self, ui: &mut Ui) {
        let mut val = self.value.value();
        ui.add(
            Slider::new(&mut val, self.minimum.value()..=self.maximum.value())
                .text(&self.name)
                .step_by(0.0001),
        );
        self.value = Signal::new(val);
    }
}

impl UiInput {
    pub fn create_node(for_input: Input, audio_buffer_len: usize) -> Arc<Node> {
        let value = for_input.default.unwrap();
        let this = Box::new(RwLock::new(UiInput {
            maximum: for_input.maximum.unwrap(),
            minimum: for_input.minimum.unwrap(),
            name: for_input.name.clone(),
            value,
        }));

        Arc::new(Node::new(
            for_input.name.clone().into(),
            audio_buffer_len,
            vec![],
            vec![Output {
                name: for_input.name.clone(),
            }],
            ProcessorType::Builtin(this),
        ))
    }
}

pub struct Constant {
    pub value: Scalar,
}

impl Constant {
    pub fn create_node(name: &str, audio_buffer_len: usize, value: Scalar) -> Arc<Node> {
        let this = Box::new(RwLock::new(Self { value }));

        Arc::new(Node::new(
            NodeName::new(name),
            audio_buffer_len,
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
    ) {
        outputs[0] = Signal::new(self.value);
    }
}

macro_rules! impl_arith {
    ($typ:ident, $op:ident, $use:ident) => {
        node_constructor! {
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
            ) {
                use std::ops::$use;
                outputs[0] = Signal::new(inputs[0].value().$op(inputs[1].value()));
            }
        }
    };
}

impl_arith!(Mul, mul, Mul);
impl_arith!(Div, div, Div);
impl_arith!(Add, add, Add);
impl_arith!(Sub, sub, Sub);
impl_arith!(Rem, rem, Rem);

node_constructor! {
    pub struct Max;
    in { a, b }
    out { out }
}

impl Processor for Max {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,

        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = Signal::new(inputs[0].value().max(inputs[1].value()));
    }
}

node_constructor! {
    pub struct Min;
    in { a, b }
    out { out }
}

impl Processor for Min {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,

        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = Signal::new(inputs[0].value().min(inputs[1].value()));
    }
}

node_constructor! {
    pub struct Abs;
    in { input }
    out { out }
}

impl Processor for Abs {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = Signal::new(inputs[0].value().abs());
    }
}

node_constructor! {
    pub struct Exp;
    in { input }
    out { out }
}

impl Processor for Exp {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = Signal::new(inputs[0].value().exp());
    }
}

node_constructor! {
    pub struct Cosine;
    in { input }
    out { out }
}

impl Processor for Cosine {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = Signal::new(inputs[0].value().cos());
    }
}

node_constructor! {
    pub struct Tanh;
    in { input }
    out { out }
}

impl Processor for Tanh {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = Signal::new(inputs[0].value().tanh());
    }
}

node_constructor! {
    pub struct Sine;
    in { input }
    out { out }
}

impl Processor for Sine {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,

        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = Signal::new(Scalar::sin(inputs[0].value()));
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
    pub fn create_nodes(name: &str, buffer_len: usize) -> (Arc<Node>, Arc<Node>) {
        let (tx, rx) = tokio::sync::watch::channel(0.0);
        let cn = Arc::new(Node::new(
            name.into(),
            buffer_len,
            vec![Input::new("c", Some(Signal::new(0.0)))],
            vec![],
            ProcessorType::Builtin(Box::new(RwLock::new(ControlToAudioTx { tx: Some(tx) }))),
        ));
        let an = Arc::new(Node::new(
            name.into(),
            buffer_len,
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
    ) {
        self.tx.as_ref().unwrap().send_replace(inputs[0].value());
    }
}
impl Processor for ControlToAudioRx {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        _inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        self.value = *self.rx.as_ref().unwrap().borrow();
        outputs[0] = Signal::new(self.value);
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
    pub fn create_nodes(name: &str, buffer_len: usize) -> (Arc<Node>, Arc<Node>) {
        let (tx, rx) = tokio::sync::watch::channel(0.0);
        let cn = Arc::new(Node::new(
            name.into(),
            buffer_len,
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
            buffer_len,
            vec![Input::new("a", Some(Signal::new(0.0)))],
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
    ) {
        self.tx.as_ref().unwrap().send_replace(inputs[0].value());
    }
}
impl Processor for AudioToControlRx {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        _inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        self.value = *self.rx.as_ref().unwrap().borrow();
        outputs[0] = Signal::new(self.value);
    }
}

node_constructor! {
    pub struct RisingEdge {
        c_last: Scalar,
    }
    in { trigger }
    out { out }
}

impl Processor for RisingEdge {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,

        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        if inputs[0].value() > self.c_last {
            outputs[0] = Signal::new(1.0);
        } else {
            outputs[0] = Signal::new(0.0);
        }
        self.c_last = inputs[0].value();
    }
}

node_constructor! {
    pub struct FallingEdge {
        c_last: Scalar,
    }
    in { trigger }
    out { out }
}

impl Processor for FallingEdge {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,

        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        if inputs[0].value() < self.c_last {
            outputs[0] = Signal::new(1.0);
        } else {
            outputs[0] = Signal::new(0.0);
        }
        self.c_last = inputs[0].value();
    }
}

node_constructor! {
    pub struct Clip;
    in { input, low, high }
    out { out }
}

impl Processor for Clip {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,

        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = Signal::new(
            inputs[0]
                .value()
                .max(inputs[1].value())
                .min(inputs[2].value()),
        );
    }
}

node_constructor! {
    pub struct If;
    in { cmp, then, els }
    out { out }
}

impl Processor for If {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        if inputs[0].value() > 0.0 {
            outputs[0] = inputs[1];
        } else {
            outputs[0] = inputs[2];
        }
    }
}

macro_rules! impl_cmp {
    ($id:ident, $op:ident) => {
        node_constructor! {
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
            ) {
                if inputs[0].$op(&inputs[1]) {
                    outputs[0] = Signal::new(1.0);
                } else {
                    outputs[0] = Signal::new(0.0);
                }
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
        node_constructor! {
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
            ) {
                if (inputs[0].value() > 0.0).$op(inputs[1].value() > 0.0) {
                    outputs[0] = Signal::new(1.0);
                } else {
                    outputs[0] = Signal::new(0.0);
                }
            }
        }
    };
}

impl_boolean!(And, bitand);
impl_boolean!(Or, bitor);
impl_boolean!(Xor, bitxor);

node_constructor! {
    pub struct Not;
    in { input }
    out { out }
}

impl Processor for Not {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        if inputs[0].value() > 0.0 {
            outputs[0] = Signal::new(0.0);
        } else {
            outputs[0] = Signal::new(1.0);
        }
    }
}
