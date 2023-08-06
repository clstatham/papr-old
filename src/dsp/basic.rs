use std::sync::{Arc, RwLock};

use eframe::egui::{Slider, Ui};
use rustc_hash::FxHashMap;

use crate::{
    dsp::SignalImpl,
    graph::{AudioNode, ControlNode, ControlOutput},
    Scalar,
};

use super::{AudioProcessor, AudioSignal, ControlProcessor, ControlSignal, SmoothControlSignal};

pub struct DummyA;
impl AudioProcessor for DummyA {
    fn process_audio(
        &mut self,
        _t: Scalar,
        _sample_rate: Scalar,
        _inputs: &FxHashMap<String, AudioSignal>,
        _control_node: &Arc<ControlNode>,
        _outputs: &mut FxHashMap<String, AudioSignal>,
    ) {
    }
}

pub struct DummyC;
impl ControlProcessor for DummyC {
    fn process_control(
        &self,
        _t: Scalar,
        _control_rate: Scalar,
        _inputs: &FxHashMap<String, ControlSignal>,
        _outputs: &FxHashMap<String, ControlOutput>,
    ) {
    }
}

pub struct DebugC {
    pub name: &'static str,
}

impl ControlProcessor for DebugC {
    fn process_control(
        &self,
        t: Scalar,
        _control_rate: Scalar,
        inputs: &FxHashMap<String, ControlSignal>,
        outputs: &FxHashMap<String, ControlOutput>,
    ) {
        println!("Debug: {} (t={t})", self.name);

        for ((inp_name, inp), (_out_name, out)) in inputs.iter().zip(outputs.iter()) {
            println!("{inp_name} = {}", inp.value());
            out.send(*inp);
        }
    }
}

pub struct Dac;

impl AudioProcessor for Dac {
    fn process_audio(
        &mut self,
        _t: Scalar,
        _sample_rate: Scalar,
        inputs: &FxHashMap<String, AudioSignal>,
        _control_node: &Arc<ControlNode>,
        outputs: &mut FxHashMap<String, AudioSignal>,
    ) {
        *outputs.get_mut("out").unwrap() = inputs["in"];
    }
}

pub struct UiInputA {
    pub value: Arc<RwLock<SmoothControlSignal>>,
}

pub struct UiInputC {
    pub name: String,
    pub minimum: ControlSignal,
    pub maximum: ControlSignal,
    pub value: Arc<RwLock<SmoothControlSignal>>,
}

impl AudioProcessor for UiInputA {
    fn process_audio(
        &mut self,
        _t: Scalar,
        _sample_rate: Scalar,
        _inputs: &FxHashMap<String, AudioSignal>,
        _control_node: &Arc<ControlNode>,
        _outputs: &mut FxHashMap<String, AudioSignal>,
    ) {
        self.value.write().unwrap().next_value();
    }
}

impl ControlProcessor for UiInputC {
    fn process_control(
        &self,
        _t: Scalar,
        _control_rate: Scalar,
        _inputs: &FxHashMap<String, ControlSignal>,
        outputs: &FxHashMap<String, ControlOutput>,
    ) {
        outputs[&self.name].send(self.value.read().unwrap().current_value());
    }

    fn ui_update(&self, ui: &mut Ui) {
        let mut val = { self.value.read().unwrap().current_value().value() };
        ui.add(Slider::new(&mut val, self.minimum.0..=self.maximum.0).text(&self.name));
        self.value.write().unwrap().set_target(ControlSignal(val));
    }
}

pub struct UiInput;
impl UiInput {
    pub fn create_nodes(
        name: &str,
        minimum: ControlSignal,
        maximum: ControlSignal,
        initial_value: ControlSignal,
    ) -> (crate::graph::AudioNode, Arc<ControlNode>) {
        let value = Arc::new(RwLock::new(SmoothControlSignal::new(initial_value, 4)));
        let cn = Arc::new(ControlNode {
            inputs: FxHashMap::default(),
            outputs: FxHashMap::from_iter(
                [(name.to_owned(), ControlOutput::new(name))].into_iter(),
            ),
            processor: Box::new(UiInputC {
                maximum,
                minimum,
                name: name.to_owned(),
                value: value.clone(),
            }),
        });
        let an = AudioNode {
            control_node: cn.clone(),
            inputs: FxHashMap::default(),
            outputs: FxHashMap::default(),
            processor: Box::new(UiInputA { value }),
        };
        (an, cn)
    }
}
