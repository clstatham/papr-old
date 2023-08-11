use std::sync::{Arc, RwLock};

use eframe::egui::{Slider, Ui};
use papr_proc_macro::node_constructor;

use crate::{
    dsp::{AudioRate, ControlRate, Signal},
    graph::{Input, Node, NodeName, Output},
    Scalar,
};

use super::{Processor, SmoothControlSignal};

pub struct Dummy;

impl Processor<AudioRate> for Dummy {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        _inputs: &[Signal<AudioRate>],
        _outputs: &mut [Signal<AudioRate>],
    ) {
    }
}

impl Processor<ControlRate> for Dummy {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<Node<AudioRate>>>,
        _inputs: &[Signal<ControlRate>],
        _outputs: &mut [Signal<ControlRate>],
    ) {
    }
}

pub struct DebugNode {
    name: String,
}

impl DebugNode {
    pub fn create_nodes(name: String) -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let this = Arc::new(RwLock::new(Self { name: name.clone() }));
        let cn = Arc::new(Node::new(
            NodeName::new(&name.clone()),
            1,
            vec![Input::new(
                "input".to_owned().as_ref(),
                Some(Signal::new(0.0)),
            )],
            vec![],
            crate::graph::ProcessorType::Builtin(this.clone()),
            None,
        ));
        let an = Arc::new(Node::new(
            NodeName::new(&name.clone()),
            0,
            vec![],
            vec![],
            crate::graph::ProcessorType::Builtin(this),
            Some(cn.clone()),
        ));
        (an, cn)
    }
}

impl Processor<AudioRate> for DebugNode {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as crate::dsp::SignalRate>::SiblingNode>>,
        _inputs: &[Signal<AudioRate>],
        _outputs: &mut [Signal<AudioRate>],
    ) {
    }
}

impl Processor<ControlRate> for DebugNode {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<AudioRate>>>,
        inputs: &[Signal<ControlRate>],
        _outputs: &mut [Signal<ControlRate>],
    ) {
        let t = inputs[1].value();
        log::debug!("{} = {} (t={t})", self.name, inputs[0].value());
    }
}

node_constructor! {
    pub struct Dac;
    @in { input }
    @out { out }
    #in {}
    #out {}
}

impl Processor<AudioRate> for Dac {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &[Signal<AudioRate>],
        outputs: &mut [Signal<AudioRate>],
    ) {
        *outputs.get_mut(0).unwrap() = inputs[0];
    }
}

impl Processor<ControlRate> for Dac {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalRate>::SiblingNode>>,
        _inputs: &[Signal<ControlRate>],
        _outputs: &mut [Signal<ControlRate>],
    ) {
    }
}
pub struct UiInput {
    pub name: String,
    pub minimum: Signal<ControlRate>,
    pub maximum: Signal<ControlRate>,
    pub value: Arc<RwLock<Signal<ControlRate>>>,
}

impl Processor<AudioRate> for UiInput {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        _inputs: &[Signal<AudioRate>],
        _outputs: &mut [Signal<AudioRate>],
    ) {
    }
}

impl Processor<ControlRate> for UiInput {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<AudioRate>>>,
        _inputs: &[Signal<ControlRate>],
        outputs: &mut [Signal<ControlRate>],
    ) {
        outputs[0] = *self.value.read().unwrap();
    }

    fn ui_update(&self, ui: &mut Ui) {
        let mut val = { self.value.read().unwrap().value() };
        ui.add(
            Slider::new(&mut val, self.minimum.0..=self.maximum.0)
                .text(&self.name)
                .step_by(0.0001),
        );
        *self.value.write().unwrap() = Signal::new_control(val);
    }
}

impl UiInput {
    pub fn create_nodes(
        for_input: Input<ControlRate>,
        audio_buffer_len: usize,
    ) -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let value = Arc::new(RwLock::new(for_input.default.unwrap()));
        let this = Arc::new(RwLock::new(UiInput {
            maximum: for_input.maximum.unwrap(),
            minimum: for_input.minimum.unwrap(),
            name: for_input.name.clone(),
            value: value.clone(),
        }));
        let cn = Arc::new(Node::new(
            for_input.name.clone().into(),
            1,
            vec![],
            vec![Output {
                name: for_input.name.clone(),
            }],
            this.clone().into(),
            None,
        ));
        let an = Arc::new(Node::new(
            for_input.name.clone().into(),
            audio_buffer_len,
            vec![],
            vec![],
            this.into(),
            Some(cn.clone()),
        ));
        (an, cn)
    }
}

pub struct Constant {
    pub value: Scalar,
}

impl Constant {
    pub fn create_nodes(
        name: &str,
        audio_buffer_len: usize,
        value: Scalar,
    ) -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let this = Arc::new(RwLock::new(Self { value }));
        let cn = Arc::new(Node::new(
            NodeName::new(name),
            1,
            vec![],
            vec![Output {
                name: "out".to_owned(),
            }],
            crate::graph::ProcessorType::Builtin(this.clone()),
            None,
        ));
        let an = Arc::new(Node::new(
            NodeName::new(name),
            audio_buffer_len,
            vec![],
            vec![Output {
                name: "out".to_owned(),
            }],
            crate::graph::ProcessorType::Builtin(this.clone()),
            Some(cn.clone()),
        ));
        (an, cn)
    }
}

impl Processor<AudioRate> for Constant {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as crate::dsp::SignalRate>::SiblingNode>>,
        _inputs: &[Signal<AudioRate>],
        outputs: &mut [Signal<AudioRate>],
    ) {
        *outputs.get_mut(0).unwrap() = self.value.into();
    }
}

impl Processor<ControlRate> for Constant {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalRate>::SiblingNode>>,
        _inputs: &[Signal<ControlRate>],
        outputs: &mut [Signal<ControlRate>],
    ) {
        *outputs.get_mut(0).unwrap() = self.value.into();
    }
}

macro_rules! impl_arith {
    ($typ:ident, $op:ident, $use:ident) => {
        impl Processor<AudioRate> for $typ {
            fn process_sample(
                &mut self,
                _buffer_idx: usize,
                _sample_rate: Scalar,
                _sibling_node: Option<&Arc<<AudioRate as super::SignalRate>::SiblingNode>>,
                inputs: &[Signal<AudioRate>],
                outputs: &mut [Signal<AudioRate>],
            ) {
                use std::ops::$use;
                *outputs.get_mut(0).unwrap() = inputs[0].$op(inputs[1]);
            }
        }

        impl Processor<ControlRate> for $typ {
            fn process_sample(
                &mut self,
                _buffer_idx: usize,
                _sample_rate: Scalar,
                _sibling_node: Option<&Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
                inputs: &[Signal<ControlRate>],
                outputs: &mut [Signal<ControlRate>],
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
    @in { a, b }
    @out { out }
    #in { a, b }
    #out { out }
}

node_constructor! {
    pub struct Divide;
    @in { a, b }
    @out { out }
    #in { a, b }
    #out { out }
}

node_constructor! {
    pub struct Add;
    @in { a, b }
    @out { out }
    #in { a, b }
    #out { out }
}

node_constructor! {
    pub struct Subtract;
    @in { a, b }
    @out { out }
    #in { a, b }
    #out { out }
}

node_constructor! {
    pub struct Sine;
    @in { input }
    @out { out }
    #in { input }
    #out { out }
}

impl Processor<AudioRate> for Sine {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as super::SignalRate>::SiblingNode>>,
        inputs: &[Signal<AudioRate>],
        outputs: &mut [Signal<AudioRate>],
    ) {
        *outputs.get_mut(0).unwrap() = Signal::new_audio(Scalar::sin(inputs[0].value()));
    }
}

impl Processor<ControlRate> for Sine {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
        inputs: &[Signal<ControlRate>],
        outputs: &mut [Signal<ControlRate>],
    ) {
        *outputs.get_mut(0).unwrap() = Signal::new_control(Scalar::sin(inputs[0].value()));
    }
}

node_constructor! {
    pub struct EventToAudio;
    @in {}
    @out { a }
    #in { e }
    #out {}
}

impl Processor<AudioRate> for EventToAudio {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        sibling_node: Option<&Arc<<AudioRate as super::SignalRate>::SiblingNode>>,
        _inputs: &[Signal<AudioRate>],
        outputs: &mut [Signal<AudioRate>],
    ) {
        let sibling_node = sibling_node.as_ref().unwrap();
        let input = sibling_node.cached_input(0).unwrap();
        *outputs.get_mut(0).unwrap() = Signal::new_audio(input.value());
    }
}

impl Processor<ControlRate> for EventToAudio {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
        _inputs: &[Signal<ControlRate>],
        _outputs: &mut [Signal<ControlRate>],
    ) {
    }
}
