use std::sync::Arc;

use crate::{
    graph::{AudioNode, AudioOutput, ControlInput, ControlNode, ControlOutput},
    Scalar, PI,
};

use super::{AudioProcessor, AudioSignal, ControlProcessor, ControlSignal, SignalImpl};

pub struct SineOsc;
pub struct SineOscC;

impl AudioProcessor for SineOsc {
    fn into_node(self) -> AudioNode {
        AudioNode {
            control_node: Arc::new(ControlNode {
                inputs: vec![
                    ControlInput::new(Some("amp"), 1.0.into()),
                    ControlInput::new(Some("freq"), 440.0.into()),
                ],
                outputs: vec![],
                processor: Box::new(SineOscC),
            }),
            inputs: vec![],
            outputs: vec![AudioOutput {
                name: Some("out".to_owned()),
            }],
            processor: Box::new(self),
        }
    }

    fn process_audio(
        &mut self,
        t: Scalar,
        _inputs: &[AudioSignal],
        control_node: &Arc<ControlNode>,
        outputs: &mut [AudioSignal],
    ) {
        let amp = control_node.read_input(0).value();
        let freq = control_node.read_input(1).value();
        outputs[0] = AudioSignal(Scalar::sin(t * PI * 2.0 * freq) * amp);
    }
}

impl ControlProcessor for SineOscC {
    fn process_control(&self, _t: Scalar, _inputs: &[ControlSignal], _outputs: &[ControlOutput]) {}
}
