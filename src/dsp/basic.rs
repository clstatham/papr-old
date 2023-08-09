use std::sync::{Arc, RwLock};

use eframe::egui::{Slider, Ui};
use papr_proc_macro::node_constructor;
use rustc_hash::FxHashMap;

use crate::{
    dsp::{AudioRate, ControlRate, Signal},
    graph::{Input, InputName, Node, NodeName, Output, OutputName},
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
        buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        _inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
    }
}

impl Processor<ControlRate> for Dummy {
    fn process_sample(
        &self,
        buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<Node<AudioRate>>>,
        _inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
    }
}

pub struct DebugNode {
    name: String,
}

impl DebugNode {
    pub fn create_nodes(name: String) -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let cn = Arc::new(Node::new(
            NodeName(name.clone()),
            1,
            FxHashMap::from_iter([(
                InputName::default(),
                Input::new(&InputName::default().0, Some(Signal::new(0.0))),
            )]),
            FxHashMap::default(),
            crate::graph::ProcessorType::Boxed(Box::new(Self { name: name.clone() })),
            None,
        ));
        let an = Arc::new(Node::new(
            NodeName(name.clone()),
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
        buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as crate::dsp::SignalType>::SiblingNode>>,
        _inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
    }
}

impl Processor<ControlRate> for DebugNode {
    fn process_sample(
        &self,
        buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<AudioRate>>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        let t = inputs[&InputName("t".to_owned())].value();
        log::debug!(
            "{} = {} (t={t})",
            self.name,
            inputs[&InputName::default()].value()
        );
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
        buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() = inputs[&InputName::default()];
    }
}

impl Processor<ControlRate> for Dac {
    fn process_sample(
        &self,
        buffer_idx: usize,
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
    fn process_sample(
        &self,
        buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<ControlRate>>>,
        _inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        self.value.write().unwrap().next_value();
    }
}

impl Processor<ControlRate> for UiInputC {
    fn process_sample(
        &self,
        buffer_idx: usize,
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
            NodeName(for_input.name.0.to_owned()),
            1,
            FxHashMap::default(),
            FxHashMap::from_iter(
                [(
                    OutputName(for_input.name.0.to_owned()),
                    Output {
                        name: OutputName(for_input.name.0.to_owned()),
                    },
                )]
                .into_iter(),
            ),
            Box::new(UiInputC {
                maximum: for_input.maximum.unwrap(),
                minimum: for_input.minimum.unwrap(),
                name: for_input.name.0.to_owned(),
                value: value.clone(),
            })
            .into(),
            None,
        ));
        let an = Arc::new(Node::new(
            NodeName(for_input.name.0.to_owned()),
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
            NodeName(name.to_owned()),
            1,
            FxHashMap::default(),
            FxHashMap::from_iter([(
                OutputName("out".to_owned()),
                Output {
                    name: OutputName("out".to_owned()),
                },
            )]),
            crate::graph::ProcessorType::Boxed(Box::new(Self { value })),
            None,
        ));
        let an = Arc::new(Node::new(
            NodeName(name.to_owned()),
            audio_buffer_len,
            FxHashMap::default(),
            FxHashMap::from_iter([(
                OutputName("out".to_owned()),
                Output {
                    name: OutputName("out".to_owned()),
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
        buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<AudioRate as crate::dsp::SignalType>::SiblingNode>>,
        _inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&OutputName("out".to_string())).unwrap() = self.value.into();
    }
}

impl Processor<ControlRate> for Constant {
    fn process_sample(
        &self,
        buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalType>::SiblingNode>>,
        _inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName("out".to_string())).unwrap() = self.value.into();
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
        buffer_idx: usize,
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
    fn process_sample(
        &self,
        buffer_idx: usize,
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
    pub struct Divide;
    @in { a, b }
    @out { out }
    #in { a, b }
    #out { out }
}

impl Processor<AudioRate> for Divide {
    fn process_sample(
        &self,
        buffer_idx: usize,
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
    fn process_sample(
        &self,
        buffer_idx: usize,
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
    pub struct Add;
    @in { a, b }
    @out { out }
    #in { a, b }
    #out { out }
}

impl Processor<AudioRate> for Add {
    fn process_sample(
        &self,
        buffer_idx: usize,
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
    fn process_sample(
        &self,
        buffer_idx: usize,
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
    pub struct Subtract;
    @in { a, b }
    @out { out }
    #in { a, b }
    #out { out }
}

impl Processor<AudioRate> for Subtract {
    fn process_sample(
        &self,
        buffer_idx: usize,
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
    fn process_sample(
        &self,
        buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() =
            inputs[&InputName("a".to_owned())] - inputs[&InputName("b".to_owned())];
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
        buffer_idx: usize,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<AudioRate as super::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() =
            Signal::new_audio(Scalar::sin(inputs[&InputName::default()].value()));
    }
}

impl Processor<ControlRate> for Sine {
    fn process_sample(
        &self,
        buffer_idx: usize,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<ControlRate as super::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        *outputs.get_mut(&OutputName::default()).unwrap() =
            Signal::new_control(Scalar::sin(inputs[&InputName::default()].value()));
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
        buffer_idx: usize,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<AudioRate as super::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        let sibling_node = sibling_node.as_ref().unwrap();
        let input = sibling_node
            .cached_input(&InputName("e".to_owned()))
            .unwrap();
        *outputs.get_mut(&OutputName("a".to_owned())).unwrap() = Signal::new_audio(input.value());
    }
}

impl Processor<ControlRate> for EventToAudio {
    fn process_sample(
        &self,
        buffer_idx: usize,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<ControlRate as super::SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
    }
}
