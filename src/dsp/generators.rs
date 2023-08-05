use std::sync::Arc;

use crate::{
    graph::{AudioNode, AudioOutput, ControlInput, ControlNode, ControlOutput, CreateNodes},
    Scalar, PI,
};

use super::{AudioProcessor, AudioSignal, ControlProcessor, ControlSignal, SignalImpl};

pub struct SineOsc;
impl CreateNodes for SineOsc {
    fn create_nodes() -> (AudioNode, Arc<ControlNode>) {
        let cn = Arc::new(ControlNode {
            inputs: vec![
                ControlInput::new(Some("amp"), 1.0.into()),
                ControlInput::new(Some("freq"), 440.0.into()),
            ],
            outputs: vec![],
            processor: Box::new(SineOscC),
        });
        let an = AudioNode {
            control_node: cn.clone(),
            inputs: vec![],
            outputs: vec![AudioOutput {
                name: Some("out".to_owned()),
            }],
            processor: Box::new(SineOscA),
        };
        (an, cn)
    }
}

pub struct SineOscA;
pub struct SineOscC;

impl AudioProcessor for SineOscA {
    fn process_audio(
        &mut self,
        t: Scalar,
        _sample_rate: Scalar,
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
    fn process_control(
        &self,
        _t: Scalar,
        _control_rate: Scalar,
        _inputs: &[ControlSignal],
        _outputs: &[ControlOutput],
    ) {
    }
}
