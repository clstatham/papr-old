use std::sync::Arc;

use crate::{
    dsp::SignalRate,
    graph::Node,
};

use super::{ParsedIdent, ParsedCreationArg};

#[derive(Debug, Clone)]
pub enum BuiltinNode {
    Sine,
    Cosine,
    Tanh,
    Exp,
    Abs,
    SineOsc,
    SineOscLFO,
    BlSawOsc,
    MidiToFreq,
    Clock,
    Delay,
    NoteIn,
    RisingEdge,
    FallingEdge,
    Var,
    Max,
    Min,
    Clip,
    Debug,
    If,
    Not,
    Sample,
    // OscReceiver,
}

impl BuiltinNode {
    pub fn try_from_ident(id: &ParsedIdent) -> Option<BuiltinNode> {
        match id.0.as_str() {
            "sin" => Some(BuiltinNode::Sine),
            "cos" => Some(BuiltinNode::Cosine),
            "exp" => Some(BuiltinNode::Exp),
            "tanh" => Some(BuiltinNode::Tanh),
            "abs" => Some(BuiltinNode::Abs),
            "sineosc" => Some(BuiltinNode::SineOsc),
            "sinelfo" => Some(BuiltinNode::SineOscLFO),
            "sawosc" => Some(BuiltinNode::BlSawOsc),
            "m2f" => Some(BuiltinNode::MidiToFreq),
            "clock" => Some(BuiltinNode::Clock),
            "delay" => Some(BuiltinNode::Delay),
            "notein" => Some(BuiltinNode::NoteIn),
            "redge" => Some(BuiltinNode::RisingEdge),
            "fedge" => Some(BuiltinNode::FallingEdge),
            "var" => Some(BuiltinNode::Var),
            "max" => Some(BuiltinNode::Max),
            "min" => Some(BuiltinNode::Min),
            "clip" => Some(BuiltinNode::Clip),
            "debug" => Some(BuiltinNode::Debug),
            "if" => Some(BuiltinNode::If),
            "not" => Some(BuiltinNode::Not),
            "sample" => Some(BuiltinNode::Sample),
            // "oscrecv" => BuiltinNode::OscReceiver),
            _ => None,
        }
    }

    pub fn create_node(
        &self,
        name: &str,
        signal_rate: SignalRate,
        audio_buffer_len: usize,
        creation_args: &[ParsedCreationArg],
    ) -> Arc<Node> {
        match self {
            Self::Debug => crate::dsp::basic::DebugNode::create_node(name, signal_rate, audio_buffer_len),
            Self::Sine => crate::dsp::basic::Sine::create_node(name, signal_rate, audio_buffer_len, 0.0),
            Self::Cosine => crate::dsp::basic::Cosine::create_node(name, signal_rate, audio_buffer_len, 0.0),
            Self::Exp => crate::dsp::basic::Exp::create_node(name, signal_rate, audio_buffer_len, 0.0),
            Self::Tanh => crate::dsp::basic::Tanh::create_node(name, signal_rate, audio_buffer_len, 0.0),
            Self::Abs => crate::dsp::basic::Abs::create_node(name, signal_rate, audio_buffer_len, 0.0),
            Self::Not => crate::dsp::basic::Not::create_node(name, signal_rate, audio_buffer_len, 0.0),
            Self::SineOsc => crate::dsp::generators::SineOsc::create_node(
                name,
                signal_rate, 
                audio_buffer_len,
                1.0,
                440.0,
                0.0,
                0.0,
            ),
            Self::SineOscLFO => {
                crate::dsp::generators::SineOscLFO::create_node(name, signal_rate, audio_buffer_len, 1.0, 1.0)
            }
            Self::BlSawOsc => crate::dsp::generators::BlSawOsc::create_node(
                name,
                signal_rate, 
                audio_buffer_len,
                0.0,
                1.0,
                0.0,
                1.0,
                440.0,
            ),
            Self::MidiToFreq => {
                crate::dsp::midi::MidiToFreq::create_node(name, signal_rate, audio_buffer_len, 0.0)
            }
            Self::Clock => crate::dsp::time::Clock::create_node(name, signal_rate, audio_buffer_len, 1.0, 0.5),
            Self::Delay => crate::dsp::time::Delay::create_node(
                name,
                signal_rate, 
                audio_buffer_len,
                vec![0.0; 480000], // TODO: don't hardcode
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ),
            Self::NoteIn => crate::io::midi::NoteIn::create_node(name, audio_buffer_len),
            Self::RisingEdge => {
                crate::dsp::basic::RisingEdge::create_node(name, signal_rate, audio_buffer_len, 0.0, 0.0)
            }
            Self::FallingEdge => {
                crate::dsp::basic::FallingEdge::create_node(name, signal_rate, audio_buffer_len, 0.0, 0.0)
            }
            Self::Var => {
                crate::dsp::graph_util::Var::create_node(name, signal_rate, audio_buffer_len, 0.0, 0.0, 0.0)
            }
            Self::Max => {
                crate::dsp::basic::Max::create_node(name, signal_rate, audio_buffer_len, 0.0, 0.0)
            }
            Self::Min => {
                crate::dsp::basic::Min::create_node(name, signal_rate, audio_buffer_len, 0.0, 0.0)
            }
            Self::Clip => {
                crate::dsp::basic::Clip::create_node(name, signal_rate, audio_buffer_len, 0.0, 0.0, 0.0)
            }
            Self::If => {
                crate::dsp::basic::If::create_node(name, signal_rate, audio_buffer_len, 0.0, 0.0, 0.0)
            }
            Self::Sample => {
                crate::dsp::samplers::Sample::create_node(name, signal_rate, audio_buffer_len, creation_args[0].clone().unwrap_string().into())
            }
            // Self::OscReceiver => crate::io::osc::OscReceiver::create_node(
            //     name,
            //     audio_buffer_len,
            //     57110,
            //     "127.0.0.1:9001",
            // ),
        }
    }
}
