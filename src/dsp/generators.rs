use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::{
    graph::{AudioNode, AudioOutput, ControlInput, ControlNode, ControlOutput, CreateNodes},
    Scalar, PI,
};

use super::{AudioProcessor, AudioSignal, ControlProcessor, ControlSignal, SignalImpl};

pub struct SineOsc;
impl CreateNodes for SineOsc {
    fn create_nodes() -> (AudioNode, Arc<ControlNode>) {
        let cn = Arc::new(ControlNode {
            inputs: FxHashMap::from_iter(
                [
                    ("amp".to_owned(), ControlInput::new("amp", 1.0.into())),
                    ("freq".to_owned(), ControlInput::new("freq", 440.0.into())),
                ]
                .into_iter(),
            ),
            outputs: FxHashMap::default(),
            processor: Box::new(SineOscC),
        });
        let an = AudioNode {
            control_node: cn.clone(),
            inputs: FxHashMap::default(),
            outputs: FxHashMap::from_iter(
                [(
                    "out".to_owned(),
                    AudioOutput {
                        name: "out".to_owned(),
                    },
                )]
                .into_iter(),
            ),
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
        _inputs: &FxHashMap<String, AudioSignal>,
        control_node: &Arc<ControlNode>,
        outputs: &mut FxHashMap<String, AudioSignal>,
    ) {
        let amp = control_node.read_input("amp").value();
        let freq = control_node.read_input("freq").value();
        *outputs.get_mut("out").unwrap() = AudioSignal(Scalar::sin(t * PI * 2.0 * freq) * amp);
    }
}

impl ControlProcessor for SineOscC {
    fn process_control(
        &self,
        _t: Scalar,
        _control_rate: Scalar,
        _inputs: &FxHashMap<String, ControlSignal>,
        _outputs: &FxHashMap<String, ControlOutput>,
    ) {
    }
}
