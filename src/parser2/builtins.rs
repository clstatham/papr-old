use nom::{
    branch::alt, bytes::complete::*, combinator::*, IResult,
};
use std::sync::Arc;

use crate::{
    dsp::SignalRate,
    graph::Node,
    Scalar,
};

use super::ParsedIdent;

pub fn global_const<'a>() -> impl FnMut(&'a str) -> IResult<&str, (String, Scalar)> {
    alt((
        value(("PI".to_owned(), crate::PI), tag("PI")),
        value(("TAU".to_owned(), crate::TAU), tag("TAU")),
        //
    ))
}

#[derive(Debug)]
pub enum BuiltinNode {
    Sine,
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
    // OscReceiver,
}

impl BuiltinNode {
    pub fn try_from_ident(id: &ParsedIdent) -> Option<BuiltinNode> {
        match id.0.as_str().strip_prefix('@').unwrap_or(id.0.as_str()) {
            "sin" => Some(BuiltinNode::Sine),
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
            // "oscrecv" => BuiltinNode::OscReceiver),
            _ => None,
        }
    }

    pub fn create_node(
        &self,
        name: &str,
        signal_rate: SignalRate,
        audio_buffer_len: usize,
    ) -> Arc<Node> {
        match self {
            Self::Debug => crate::dsp::basic::DebugNode::create_node(name, signal_rate, audio_buffer_len),
            Self::Sine => crate::dsp::basic::Sine::create_node(name, signal_rate, audio_buffer_len, 0.0),
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
            // Self::OscReceiver => crate::io::osc::OscReceiver::create_node(
            //     name,
            //     audio_buffer_len,
            //     57110,
            //     "127.0.0.1:9001",
            // ),
        }
    }
}
