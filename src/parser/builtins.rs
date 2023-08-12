use nom::{
    branch::alt, bytes::complete::*, combinator::*, multi::*, number::complete::float, sequence::*,
    IResult,
};
use std::sync::{Arc, Mutex};

use crate::{
    dsp::SignalRate,
    graph::{Node, NodeName},
    Scalar,
};

use super::{ident, ignore_garbage, whitespace1, CreateRhs, ParsedCreate};

pub fn global_const<'a>() -> impl FnMut(&'a str) -> IResult<&str, (String, Scalar)> {
    alt((
        value(("PI".to_owned(), crate::PI), tag("PI")),
        value(("TAU".to_owned(), crate::TAU), tag("TAU")),
        //
    ))
}

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
    // OscReceiver,
}

impl BuiltinNode {
    pub fn create_node(
        &self,
        name: &str,
        signal_rate: SignalRate,
        audio_buffer_len: usize,
    ) -> Arc<Node> {
        match self {
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

pub fn create_statement<'a>() -> impl FnMut(&'a str) -> IResult<&str, ParsedCreate> {
    map(
        tuple((
            tag("create"),
            whitespace1(),
            alt((value(SignalRate::Audio, tag("@")), value(SignalRate::Control, tag("#")))),
            ident(),
            ignore_garbage(tag(":")),
            ident(),
            opt(delimited(
                ignore_garbage(tag("<")),
                many1(ignore_garbage(float)),
                ignore_garbage(tag(">")),
            )),
            ignore_garbage(tag(";")),
        )),
        // todo: defaults are broken
        |(_, _, signal_rate, id, _, graph_name, _control_input_defaults, _)| ParsedCreate {
            
            ident: id.to_owned(),
            signal_rate,
            rhs: match graph_name {
                "sin" => CreateRhs::BuiltinNode(BuiltinNode::Sine),
                "sineosc" => CreateRhs::BuiltinNode(BuiltinNode::SineOsc),
                "sinelfo" => CreateRhs::BuiltinNode(BuiltinNode::SineOscLFO),
                "sawosc" => CreateRhs::BuiltinNode(BuiltinNode::BlSawOsc),
                "m2f" => CreateRhs::BuiltinNode(BuiltinNode::MidiToFreq),
                "clock" => CreateRhs::BuiltinNode(BuiltinNode::Clock),
                "delay" => CreateRhs::BuiltinNode(BuiltinNode::Delay),
                "notein" => CreateRhs::BuiltinNode(BuiltinNode::NoteIn),
                "redge" => CreateRhs::BuiltinNode(BuiltinNode::RisingEdge),
                "fedge" => CreateRhs::BuiltinNode(BuiltinNode::FallingEdge),
                "var" => CreateRhs::BuiltinNode(BuiltinNode::Var),
                "max" => CreateRhs::BuiltinNode(BuiltinNode::Max),
                "min" => CreateRhs::BuiltinNode(BuiltinNode::Min),
                "clip" => CreateRhs::BuiltinNode(BuiltinNode::Clip),
                // "oscrecv" => CreateRhs::BuiltinNode(BuiltinNode::OscReceiver),
                _ => CreateRhs::ScriptGraph(NodeName::new(graph_name)),
            },
        },
    )
}
