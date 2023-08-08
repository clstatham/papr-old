use std::sync::{Arc, RwLock};

use eframe::egui::{Slider, Ui};
use rustc_hash::FxHashMap;

use crate::{
    dsp::{AudioRate, ControlRate, Signal},
    graph::{InputName, Node, NodeName, Output, OutputName},
    node_constructor, Scalar,
};

use super::{Processor, SmoothControlSignal};

node_constructor! {
    Dummy {}
    @in {}
    @out {}
    #in {}
    #out {}
}

impl Processor<AudioRate> for Dummy {
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

impl Processor<ControlRate> for Dummy {
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

node_constructor! {
    DebugNode {name_copy: String}
    @in {}
    @out {}
    #in { "in" = 0.0 }
    #out {}
}

// pub struct DebugNode;

impl Processor<AudioRate> for DebugNode {
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

impl Processor<ControlRate> for DebugNode {
    fn process(
        &self,
        t: Scalar,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<AudioRate>>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        log::debug!(
            "{} = {} (t={t})",
            self.name_copy,
            inputs[&InputName::default()].value()
        );
    }
}

node_constructor! {
    Dac {}
    @in { "in" = 0.0 }
    @out { "out" }
    #in {}
    #out {}
}

impl Processor<AudioRate> for Dac {
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

impl Processor<ControlRate> for Dac {
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
            NodeName(name.to_owned()),
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
            NodeName(name.to_owned()),
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
