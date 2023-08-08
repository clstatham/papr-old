use std::sync::{Arc, RwLock};

use eframe::egui::{Slider, Ui};
use rustc_hash::FxHashMap;

use crate::{
    dsp::{AudioRate, ControlRate, Signal},
    graph::{CreateNodes, Input, InputName, Node, Output, OutputName},
    node_constructor, Scalar,
};

use super::{Processor, SmoothControlSignal};

pub struct DummyA;
impl Processor<AudioRate> for DummyA {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        _inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
    }
}

pub struct DummyC;
impl Processor<ControlRate> for DummyC {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<Node<AudioRate>>>,
        _inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
    }
}

pub struct DebugNode;

pub struct DebugNodeA;

pub struct DebugNodeC {
    pub name: String,
}

impl Processor<AudioRate> for DebugNodeA {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as crate::dsp::SignalType>::SiblingNode>>,
        _inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
    }
}

impl Processor<ControlRate> for DebugNodeC {
    fn process(
        &self,
        t: Scalar,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<AudioRate>>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        println!(
            "Debug: {} (t={t}) => {}",
            self.name,
            inputs[&InputName::default()].value()
        );
    }
}

impl DebugNode {
    pub fn create_nodes(name: &str) -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let cn = Arc::new(Node::new(
            FxHashMap::from_iter(
                [(
                    InputName::default(),
                    Input::new("in", Signal::new_control(0.0)),
                )]
                .into_iter(),
            ),
            FxHashMap::default(),
            Box::new(DebugNodeC {
                name: name.to_owned(),
            })
            .into(),
            None,
        ));
        let an = Arc::new(Node::new(
            FxHashMap::default(),
            FxHashMap::default(),
            Box::new(DebugNodeA).into(),
            Some(cn.clone()),
        ));
        (an, cn)
    }
}

pub struct Dac;
pub struct DacA;
pub struct DacC;

impl Processor<AudioRate> for DacA {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() = inputs[&InputName::default()];
    }
}

impl Processor<ControlRate> for DacC {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalType>::SiblingNode>>,
        _inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
    }
}

impl CreateNodes for Dac {
    fn create_nodes() -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let cn = Arc::new(Node::new(
            FxHashMap::default(),
            FxHashMap::default(),
            Box::new(DacC).into(),
            None,
        ));
        let an = Arc::new(Node::new(
            FxHashMap::from_iter(
                [(
                    InputName::default(),
                    Input {
                        name: InputName::default(),
                        default: Signal::new(0.0),
                    },
                )]
                .into_iter(),
            ),
            FxHashMap::from_iter(
                [(
                    OutputName::default(),
                    Output {
                        name: OutputName::default(),
                    },
                )]
                .into_iter(),
            ),
            Box::new(DacA).into(),
            Some(cn.clone()),
        ));
        (an, cn)
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
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        _inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        self.value.write().unwrap().next_value();
    }
}

impl Processor<ControlRate> for UiInputC {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<AudioRate>>>,
        _inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName(self.name.clone())).unwrap() =
            self.value.read().unwrap().current_value();
    }

    fn ui_update(&self, ui: &mut Ui) {
        let mut val = { self.value.read().unwrap().current_value().value() };
        ui.add(Slider::new(&mut val, self.minimum.0..=self.maximum.0).text(&self.name));
        self.value
            .write()
            .unwrap()
            .set_target(Signal::new_control(val));
    }
}

pub struct UiInput;
impl UiInput {
    pub fn create_nodes(
        name: &str,
        minimum: Signal<ControlRate>,
        maximum: Signal<ControlRate>,
        initial_value: Signal<ControlRate>,
    ) -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let value = Arc::new(RwLock::new(SmoothControlSignal::new(initial_value, 4)));
        let cn = Arc::new(Node::new(
            FxHashMap::default(),
            FxHashMap::from_iter(
                [(
                    OutputName(name.to_owned()),
                    Output {
                        name: OutputName(name.to_owned()),
                    },
                )]
                .into_iter(),
            ),
            Box::new(UiInputC {
                maximum,
                minimum,
                name: name.to_owned(),
                value: value.clone(),
            })
            .into(),
            None,
        ));
        let an = Arc::new(Node::new(
            FxHashMap::default(),
            FxHashMap::default(),
            Box::new(UiInputA { value }).into(),
            Some(cn.clone()),
        ));
        (an, cn)
    }
}

node_constructor! {
    Constant {
        value: Scalar
    }
    @in {}
    @out { "out" }
    #in {}
    #out { "out" }
}

impl Processor<AudioRate> for Constant {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as crate::dsp::SignalType>::SiblingNode>>,
        _inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&OutputName("out".to_string())).unwrap() = self.value.into();
    }
}

impl Processor<ControlRate> for Constant {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalType>::SiblingNode>>,
        _inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName("out".to_string())).unwrap() = self.value.into();
    }
}

node_constructor! {
    Multiply {}
    @in { "a" = 0.0 "b" = 0.0 }
    @out { "out" }
    #in { "a" = 0.0 "b" = 0.0 }
    #out { "out" }
}

impl Processor<AudioRate> for Multiply {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as super::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() =
            inputs[&InputName("a".to_owned())] * inputs[&InputName("b".to_owned())];
    }
}

impl Processor<ControlRate> for Multiply {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() =
            inputs[&InputName("a".to_owned())] * inputs[&InputName("b".to_owned())];
    }
}

node_constructor! {
    Divide {}
    @in { "a" = 0.0 "b" = 0.0 }
    @out { "out" }
    #in { "a" = 0.0 "b" = 0.0 }
    #out { "out" }
}

impl Processor<AudioRate> for Divide {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as super::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() =
            inputs[&InputName("a".to_owned())] / inputs[&InputName("b".to_owned())];
    }
}

impl Processor<ControlRate> for Divide {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() =
            inputs[&InputName("a".to_owned())] / inputs[&InputName("b".to_owned())];
    }
}

node_constructor! {
    Add {}
    @in { "a" = 0.0 "b" = 0.0 }
    @out { "out" }
    #in { "a" = 0.0 "b" = 0.0 }
    #out { "out" }
}

impl Processor<AudioRate> for Add {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as super::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() =
            inputs[&InputName("a".to_owned())] + inputs[&InputName("b".to_owned())];
    }
}

impl Processor<ControlRate> for Add {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() =
            inputs[&InputName("a".to_owned())] + inputs[&InputName("b".to_owned())];
    }
}

node_constructor! {
    Subtract {}
    @in { "a" = 0.0 "b" = 0.0 }
    @out { "out" }
    #in { "a" = 0.0 "b" = 0.0 }
    #out { "out" }
}

impl Processor<AudioRate> for Subtract {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as super::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() =
            inputs[&InputName("a".to_owned())] - inputs[&InputName("b".to_owned())];
    }
}

impl Processor<ControlRate> for Subtract {
    fn process(
        &self,
        _t: Scalar,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() =
            inputs[&InputName("a".to_owned())] - inputs[&InputName("b".to_owned())];
    }
}
