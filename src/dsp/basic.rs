use std::sync::{Arc, RwLock};

use eframe::egui::{Slider, Ui};
use rustc_hash::FxHashMap;

use crate::{
    dsp::Signal,
    graph::{AudioRate, ControlRate, CreateNodes, Input, InputName, Node, Output, OutputName},
    Scalar,
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

pub struct DebugC {
    pub name: &'static str,
}

impl Processor<ControlRate> for DebugC {
    fn process(
        &self,
        t: Scalar,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<AudioRate>>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        println!("Debug: {} (t={t})", self.name);

        for ((inp_name, inp), (_out_name, _out)) in inputs.iter().zip(outputs.iter()) {
            println!("{inp_name} = {}", inp.value());
        }
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
        *outputs.get_mut(&OutputName("out".to_owned())).unwrap() =
            inputs[&InputName("in".to_owned())];
    }
}

impl Processor<ControlRate> for DacC {
    fn process(
        &self,
        t: Scalar,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<ControlRate as crate::graph::GraphKind>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
    }
}

impl CreateNodes for Dac {
    fn create_nodes() -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let cn = Arc::new(Node::new(
            FxHashMap::default(),
            FxHashMap::default(),
            Box::new(DacC),
            None,
        ));
        let an = Arc::new(Node::new(
            FxHashMap::from_iter(
                [(
                    InputName("in".to_owned()),
                    Input {
                        name: InputName("in".to_owned()),
                        default: Signal::new(0.0),
                    },
                )]
                .into_iter(),
            ),
            FxHashMap::from_iter(
                [(
                    OutputName("out".to_owned()),
                    Output {
                        name: OutputName("out".to_owned()),
                    },
                )]
                .into_iter(),
            ),
            Box::new(DacA),
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
            }),
            None,
        ));
        let an = Arc::new(Node::new(
            FxHashMap::default(),
            FxHashMap::default(),
            Box::new(UiInputA { value }),
            Some(cn.clone()),
        ));
        (an, cn)
    }
}
