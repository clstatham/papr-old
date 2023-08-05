use std::sync::{Arc, RwLock};

use eframe::egui::{Slider, Ui};

use crate::{
    dsp::SignalImpl,
    graph::{ControlNode, ControlOutput},
    Scalar,
};

use super::{AudioProcessor, AudioSignal, ControlProcessor, ControlSignal};

pub struct DummyC;
impl ControlProcessor for DummyC {
    fn process_control(&self, _t: Scalar, _inputs: &[ControlSignal], _outputs: &[ControlOutput]) {}
}

pub struct DebugC {
    pub name: &'static str,
}

impl ControlProcessor for DebugC {
    fn process_control(&self, t: Scalar, inputs: &[ControlSignal], outputs: &[ControlOutput]) {
        println!("Debug: {} (t={t})", self.name);

        for (inp, out) in inputs.iter().zip(outputs.iter()) {
            println!("{}", inp.value());
            out.tx.send_replace(*inp);
        }
    }
}

pub struct Dac;

impl AudioProcessor for Dac {
    fn process_audio(
        &mut self,
        _t: Scalar,
        inputs: &[AudioSignal],
        _control_node: &Arc<ControlNode>,
        outputs: &mut [AudioSignal],
    ) {
        outputs.copy_from_slice(inputs);
    }
}

pub struct UiInputC {
    pub name: String,
    pub minimum: ControlSignal,
    pub maximum: ControlSignal,
    pub value: RwLock<ControlSignal>,
}

impl ControlProcessor for UiInputC {
    fn process_control(&self, t: Scalar, inputs: &[ControlSignal], outputs: &[ControlOutput]) {
        outputs[0].send(*self.value.read().unwrap());
    }

    fn ui_update(&self, ui: &mut Ui) {
        ui.add(
            Slider::new(
                &mut self.value.write().unwrap().0,
                self.minimum.0..=self.maximum.0,
            )
            .text(&self.name),
        );
    }
}
