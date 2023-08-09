use std::sync::{Arc, Mutex};

use rustc_hash::FxHashMap;

use crate::{
    dsp::{AudioRate, ControlRate},
    graph::{InputName, Node, OutputName},
    node_constructor, Scalar, PI, TAU,
};

use super::{Processor, Signal};

node_constructor! {
    pub struct SineOsc;
    @in { fm }
    @out { out }
    #in { amp, freq, fm_amt }
    #out {}
}

impl Processor<AudioRate> for SineOsc {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        sibling_node: Option<&Arc<Node<ControlRate>>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        let sibling_node = sibling_node.as_ref().unwrap();
        let amp = sibling_node
            .cached_input(&InputName("amp".to_owned()))
            .unwrap();
        let freq = sibling_node
            .cached_input(&InputName("freq".to_owned()))
            .unwrap();
        let fm_amt = sibling_node
            .cached_input(&InputName("fm_amt".to_owned()))
            .unwrap();
        let t = inputs[&InputName("t".to_owned())].value();
        let fm = inputs[&InputName("fm".to_owned())];
        *outputs.get_mut(&OutputName::default()).unwrap() = Signal::new_audio(
            Scalar::sin(t * TAU * freq.value() + fm.value() * TAU * fm_amt.value()) * amp.value(),
        );
    }
}

impl Processor<ControlRate> for SineOsc {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _control_node: Option<&Arc<Node<AudioRate>>>,
        _inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
    }
}

node_constructor! {
    pub struct BlSawOsc {
        p: Arc<Mutex<Scalar>>,
        /// SET TO 1.0 INITIALLY
        dp: Arc<Mutex<Scalar>>,
        saw: Arc<Mutex<Scalar>>,
    }
    @in {}
    @out { out }
    #in { amp, freq }
    #out {}
}

impl Processor<AudioRate> for BlSawOsc {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<AudioRate as super::SignalType>::SiblingNode>>,
        _inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        let sibling_node = sibling_node.as_ref().unwrap();
        let amp = sibling_node
            .cached_input(&InputName("amp".to_owned()))
            .unwrap();
        let freq = sibling_node
            .cached_input(&InputName("freq".to_owned()))
            .unwrap();
        let mut saw = self.saw.lock().unwrap();
        let mut p = self.p.lock().unwrap();
        let mut dp = self.dp.lock().unwrap();

        // algorithm courtesy of https://www.musicdsp.org/en/latest/Synthesis/12-bandlimited-waveforms.html

        let pmax = 0.5 * sample_rate / freq.value();
        let dc = -0.498 / pmax;

        *p += *dp;
        if *p < 0.0 {
            *p = -*p;
            *dp = -*dp;
        } else if *p > pmax {
            *p = pmax + pmax - *p;
            *dp = -*dp;
        }
        let mut x = PI * *p;
        if x < 0.00001 {
            x = 0.00001;
        }
        *saw = 0.995 * *saw + dc + x.sin() / x;

        *outputs.get_mut(&OutputName::default()).unwrap() = Signal::new_audio(*saw * amp.value());
    }
}

impl Processor<ControlRate> for BlSawOsc {
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<ControlRate as super::SignalType>::SiblingNode>>,
        _inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        _outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
    }
}
