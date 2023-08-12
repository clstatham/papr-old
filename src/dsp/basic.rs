use std::sync::{Arc, RwLock};

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
    pub fn create_node(name: String, signal_rate: SignalRate, buffer_len: usize) -> Arc<Node> {
        let this = Box::new(RwLock::new(Self { name: name.clone() }));

        Arc::new(Node::new(
            NodeName::new(&name.clone()),
            signal_rate,
            buffer_len,
            vec![Input::new(
                "input".to_owned().as_ref(),
                Some(Signal::new(0.0)),
            )],
            vec![],
            crate::graph::ProcessorType::Builtin(this),
        ))
    }
}

impl Processor for DebugNode {
    fn process_control_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        inputs: &[Signal],
        _outputs: &mut [Signal],
    ) {
        let t = inputs[1].value();
        log::debug!("{} = {} (t={t})", self.name, inputs[0].value());
    }
}

pub struct UiInput {
    pub name: String,
    pub minimum: Signal,
    pub maximum: Signal,
    pub value: Arc<RwLock<Signal>>,
}

impl Processor for UiInput {
    fn process_control_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = *self.value.read().unwrap();
    }

    fn ui_update(&self, ui: &mut Ui) {
        let mut val = { self.value.read().unwrap().value() };
        ui.add(
            Slider::new(&mut val, self.minimum.value()..=self.maximum.value())
                .text(&self.name)
                .step_by(0.0001),
        );
        *self.value.write().unwrap() = Signal::new(val);
    }
}

impl UiInput {
    pub fn create_node(for_input: Input, audio_buffer_len: usize) -> Arc<Node> {
        let value = Arc::new(RwLock::new(for_input.default.unwrap()));
        let this = Box::new(RwLock::new(UiInput {
            maximum: for_input.maximum.unwrap(),
            minimum: for_input.minimum.unwrap(),
            name: for_input.name.clone(),
            value: value.clone(),
        }));

        Arc::new(Node::new(
            for_input.name.clone().into(),
            SignalRate::Control,
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
    pub fn create_node(
        name: &str,
        signal_rate: SignalRate,
        audio_buffer_len: usize,
        value: Scalar,
    ) -> Arc<Node> {
        let this = Box::new(RwLock::new(Self { value }));

        Arc::new(Node::new(
            NodeName::new(name),
            signal_rate,
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
    fn process_audio_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        *outputs.get_mut(0).unwrap() = Signal::new(self.value);
    }

    fn process_control_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        *outputs.get_mut(0).unwrap() = Signal::new(self.value);
    }
}

macro_rules! impl_arith {
    ($typ:ident, $op:ident, $use:ident) => {
        impl Processor for $typ {
            fn process_audio_sample(
                &mut self,
                _buffer_idx: usize,
                _sample_rate: Scalar,
                inputs: &[Signal],
                outputs: &mut [Signal],
            ) {
                use std::ops::$use;
                *outputs.get_mut(0).unwrap() = inputs[0].$op(inputs[1]);
            }

            fn process_control_sample(
                &mut self,
                _buffer_idx: usize,
                _sample_rate: Scalar,
                inputs: &[Signal],
                outputs: &mut [Signal],
            ) {
                use std::ops::$use;
                *outputs.get_mut(0).unwrap() = inputs[0].$op(inputs[1]);
            }
        }
    };
}

impl_arith!(Multiply, mul, Mul);
impl_arith!(Divide, div, Div);
impl_arith!(Add, add, Add);
impl_arith!(Subtract, sub, Sub);

node_constructor! {
    pub struct Multiply;
    in { a, b }
    out { out }
}

node_constructor! {
    pub struct Divide;
    in { a, b }
    out { out }
}

node_constructor! {
    pub struct Add;
    in { a, b }
    out { out }
}

node_constructor! {
    pub struct Subtract;
    in { a, b }
    out { out }
}

node_constructor! {
    pub struct Max;
    in { a, b }
    out { out }
}

impl Processor for Max {
    fn process_audio_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,

        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = Signal::new(inputs[0].value().max(inputs[1].value()));
    }

    fn process_control_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,

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
    fn process_audio_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,

        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = Signal::new(inputs[0].value().min(inputs[1].value()));
    }

    fn process_control_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,

        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        outputs[0] = Signal::new(inputs[0].value().min(inputs[1].value()));
    }
}

node_constructor! {
    pub struct Sine;
    in { input }
    out { out }
}

impl Processor for Sine {
    fn process_audio_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,

        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        *outputs.get_mut(0).unwrap() = Signal::new(Scalar::sin(inputs[0].value()));
    }

    fn process_control_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,

        inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        *outputs.get_mut(0).unwrap() = Signal::new(Scalar::sin(inputs[0].value()));
    }
}

pub struct ControlToAudio {
    tx: Option<tokio::sync::watch::Sender<Scalar>>,
    rx: Option<tokio::sync::watch::Receiver<Scalar>>,
    value: Scalar,
}

impl ControlToAudio {
    pub fn create_nodes(name: &str, buffer_len: usize) -> (Arc<Node>, Arc<Node>) {
        let (tx, rx) = tokio::sync::watch::channel(0.0);
        let cn = Arc::new(Node::new(
            name.into(),
            SignalRate::Control,
            buffer_len,
            vec![Input::new("c", Some(Signal::new(0.0)))],
            vec![],
            ProcessorType::Builtin(Box::new(RwLock::new(Self {
                tx: Some(tx),
                rx: None,
                value: 0.0,
            }))),
        ));
        let an = Arc::new(Node::new(
            name.into(),
            SignalRate::Audio,
            buffer_len,
            vec![],
            vec![Output {
                name: "a".to_owned(),
            }],
            ProcessorType::Builtin(Box::new(RwLock::new(Self {
                tx: None,
                rx: Some(rx),
                value: 0.0,
            }))),
        ));
        (an, cn)
    }
}

impl Processor for ControlToAudio {
    fn process_control_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        inputs: &[Signal],
        _outputs: &mut [Signal],
    ) {
        self.tx.as_ref().unwrap().send_replace(inputs[0].value());
    }

    fn process_audio_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
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
    fn process_control_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,

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
    fn process_control_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,

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
    fn process_control_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,

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
