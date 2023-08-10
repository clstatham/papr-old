use std::sync::{Arc, RwLock};

use eframe::egui::{Slider, Ui};
use papr_proc_macro::node_constructor;
use rustc_hash::FxHashMap;

use crate::{
    dsp::{AudioRate, ControlRate, Signal},
    graph::{Input, Node, NodeName, Output},
    Scalar,
};

use super::{Processor, SmoothControlSignal};

node_constructor! {
    pub struct Dummy;
    @in {}
    @out {}
    #in {}
    #out {}
}

impl Processor<AudioRate> for Dummy {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        _inputs: &FxHashMap<&str, Signal<AudioRate>>,
        _outputs: &mut FxHashMap<&str, Signal<AudioRate>>,
    ) {
    }
}

impl Processor<ControlRate> for Dummy {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<Node<AudioRate>>>,
        _inputs: &FxHashMap<&str, Signal<ControlRate>>,
        _outputs: &mut FxHashMap<&str, Signal<ControlRate>>,
    ) {
    }
}

pub struct DebugNode {
    name: String,
}

impl DebugNode {
    pub fn create_nodes(name: String) -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let cn = Arc::new(Node::new(
            NodeName::new(&name.clone()),
            1,
            FxHashMap::from_iter([(
                "input".to_owned(),
                Input::new("input".to_owned().as_ref(), Some(Signal::new(0.0))),
            )]),
            FxHashMap::default(),
            crate::graph::ProcessorType::Boxed(Box::new(Self { name: name.clone() })),
            None,
        ));
        let an = Arc::new(Node::new(
            NodeName::new(&name.clone()),
            0,
            FxHashMap::default(),
            FxHashMap::default(),
            crate::graph::ProcessorType::Boxed(Box::new(Self { name: name.clone() })),
            Some(cn.clone()),
        ));
        (an, cn)
    }
}

impl Processor<AudioRate> for DebugNode {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as crate::dsp::SignalRate>::SiblingNode>>,
        _inputs: &FxHashMap<&str, Signal<AudioRate>>,
        _outputs: &mut FxHashMap<&str, Signal<AudioRate>>,
    ) {
    }
}

impl Processor<ControlRate> for DebugNode {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<AudioRate>>>,
        inputs: &FxHashMap<&str, Signal<ControlRate>>,
        _outputs: &mut FxHashMap<&str, Signal<ControlRate>>,
    ) {
        let t = inputs["t"].value();
        log::debug!("{} = {} (t={t})", self.name, inputs["input"].value());
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
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &FxHashMap<&str, Signal<AudioRate>>,
        outputs: &mut FxHashMap<&str, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&"out").unwrap() = inputs["input"];
    }
}

impl Processor<ControlRate> for Dac {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalRate>::SiblingNode>>,
        _inputs: &FxHashMap<&str, Signal<ControlRate>>,
        _outputs: &mut FxHashMap<&str, Signal<ControlRate>>,
    ) {
    }
}

pub struct UiInputA {
    pub value: Arc<RwLock<SmoothControlSignal>>,
}

pub struct UiInputC {
    pub name: String,
    pub minimum: Signal<ControlRate>,
    pub maximum: Signal<ControlRate>,
    pub value: Arc<RwLock<SmoothControlSignal>>,
}

impl Processor<AudioRate> for UiInputA {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        _inputs: &FxHashMap<&str, Signal<AudioRate>>,
        _outputs: &mut FxHashMap<&str, Signal<AudioRate>>,
    ) {
        self.value.write().unwrap().next_value();
    }
}

impl Processor<ControlRate> for UiInputC {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<AudioRate>>>,
        _inputs: &FxHashMap<&str, Signal<ControlRate>>,
        outputs: &mut FxHashMap<&str, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(self.name.as_str()).unwrap() = self.value.read().unwrap().current_value();
    }

    fn ui_update(&self, ui: &mut Ui) {
        let mut val = { self.value.read().unwrap().current_value().value() };
        ui.add(
            Slider::new(&mut val, self.minimum.0..=self.maximum.0)
                .text(&self.name)
                .step_by(0.0001),
        );
        self.value
            .write()
            .unwrap()
            .set_target(Signal::new_control(val));
    }
}

pub struct UiInput;
impl UiInput {
    pub fn create_nodes(
        for_input: Input<ControlRate>,
        audio_buffer_len: usize,
    ) -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let value = Arc::new(RwLock::new(SmoothControlSignal::new(
            for_input.default.unwrap(),
            4,
        )));
        let cn = Arc::new(Node::new(
            for_input.name.clone().into(),
            1,
            FxHashMap::default(),
            FxHashMap::from_iter(
                [(
                    for_input.name.clone(),
                    Output {
                        name: for_input.name.clone(),
                    },
                )]
                .into_iter(),
            ),
            Box::new(UiInputC {
                maximum: for_input.maximum.unwrap(),
                minimum: for_input.minimum.unwrap(),
                name: for_input.name.clone(),
                value: value.clone(),
            })
            .into(),
            None,
        ));
        let an = Arc::new(Node::new(
            for_input.name.clone().into(),
            audio_buffer_len,
            FxHashMap::default(),
            FxHashMap::default(),
            Box::new(UiInputA { value }).into(),
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
        let cn = Arc::new(Node::new(
            NodeName::new(name),
            1,
            FxHashMap::default(),
            FxHashMap::from_iter([(
                "out".to_owned(),
                Output {
                    name: "out".to_owned(),
                },
            )]),
            crate::graph::ProcessorType::Boxed(Box::new(Self { value })),
            None,
        ));
        let an = Arc::new(Node::new(
            NodeName::new(name),
            audio_buffer_len,
            FxHashMap::default(),
            FxHashMap::from_iter([(
                "out".to_owned(),
                Output {
                    name: "out".to_owned(),
                },
            )]),
            crate::graph::ProcessorType::Boxed(Box::new(Self { value })),
            Some(cn.clone()),
        ));
        (an, cn)
    }
}

impl Processor<AudioRate> for Constant {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as crate::dsp::SignalRate>::SiblingNode>>,
        _inputs: &FxHashMap<&str, Signal<AudioRate>>,
        outputs: &mut FxHashMap<&str, Signal<AudioRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = self.value.into();
    }
}

impl Processor<ControlRate> for Constant {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalRate>::SiblingNode>>,
        _inputs: &FxHashMap<&str, Signal<ControlRate>>,
        outputs: &mut FxHashMap<&str, Signal<ControlRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = self.value.into();
    }
}

node_constructor! {
    pub struct Multiply;
    @in { a, b }
    @out { out }
    #in { a, b }
    #out { out }
}

impl Processor<AudioRate> for Multiply {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as super::SignalRate>::SiblingNode>>,
        inputs: &FxHashMap<&str, Signal<AudioRate>>,
        outputs: &mut FxHashMap<&str, Signal<AudioRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = inputs["a"] * inputs["b"];
    }
}

impl Processor<ControlRate> for Multiply {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
        inputs: &FxHashMap<&str, Signal<ControlRate>>,
        outputs: &mut FxHashMap<&str, Signal<ControlRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = inputs["a"] * inputs["b"];
    }
}

node_constructor! {
    pub struct Divide;
    @in { a, b }
    @out { out }
    #in { a, b }
    #out { out }
}

impl Processor<AudioRate> for Divide {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as super::SignalRate>::SiblingNode>>,
        inputs: &FxHashMap<&str, Signal<AudioRate>>,
        outputs: &mut FxHashMap<&str, Signal<AudioRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = inputs["a"] / inputs["b"];
    }
}

impl Processor<ControlRate> for Divide {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
        inputs: &FxHashMap<&str, Signal<ControlRate>>,
        outputs: &mut FxHashMap<&str, Signal<ControlRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = inputs["a"] / inputs["b"];
    }
}

node_constructor! {
    pub struct Add;
    @in { a, b }
    @out { out }
    #in { a, b }
    #out { out }
}

impl Processor<AudioRate> for Add {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as super::SignalRate>::SiblingNode>>,
        inputs: &FxHashMap<&str, Signal<AudioRate>>,
        outputs: &mut FxHashMap<&str, Signal<AudioRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = inputs["a"] + inputs["b"];
    }
}

impl Processor<ControlRate> for Add {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
        inputs: &FxHashMap<&str, Signal<ControlRate>>,
        outputs: &mut FxHashMap<&str, Signal<ControlRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = inputs["a"] + inputs["b"];
    }
}

node_constructor! {
    pub struct Subtract;
    @in { a, b }
    @out { out }
    #in { a, b }
    #out { out }
}

impl Processor<AudioRate> for Subtract {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as super::SignalRate>::SiblingNode>>,
        inputs: &FxHashMap<&str, Signal<AudioRate>>,
        outputs: &mut FxHashMap<&str, Signal<AudioRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = inputs["a"] - inputs["b"];
    }
}

impl Processor<ControlRate> for Subtract {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
        inputs: &FxHashMap<&str, Signal<ControlRate>>,
        outputs: &mut FxHashMap<&str, Signal<ControlRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = inputs["a"] - inputs["b"];
    }
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
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as super::SignalRate>::SiblingNode>>,
        inputs: &FxHashMap<&str, Signal<AudioRate>>,
        outputs: &mut FxHashMap<&str, Signal<AudioRate>>,
    ) {
        *outputs.get_mut("out").unwrap() = Signal::new_audio(Scalar::sin(inputs["input"].value()));
    }
}

impl Processor<ControlRate> for Sine {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
        inputs: &FxHashMap<&str, Signal<ControlRate>>,
        outputs: &mut FxHashMap<&str, Signal<ControlRate>>,
    ) {
        *outputs.get_mut("out").unwrap() =
            Signal::new_control(Scalar::sin(inputs["input"].value()));
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
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        sibling_node: Option<&Arc<<AudioRate as super::SignalRate>::SiblingNode>>,
        _inputs: &FxHashMap<&str, Signal<AudioRate>>,
        outputs: &mut FxHashMap<&str, Signal<AudioRate>>,
    ) {
        let sibling_node = sibling_node.as_ref().unwrap();
        let input = sibling_node.cached_input("e").unwrap();
        *outputs.get_mut("a").unwrap() = Signal::new_audio(input.value());
    }
}

impl Processor<ControlRate> for EventToAudio {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalRate>::SiblingNode>>,
        _inputs: &FxHashMap<&str, Signal<ControlRate>>,
        _outputs: &mut FxHashMap<&str, Signal<ControlRate>>,
    ) {
    }
}
