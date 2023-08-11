use nom::{
    branch::alt, bytes::complete::*, combinator::*, multi::*, number::complete::float, sequence::*,
    IResult,
};
use std::sync::{Arc, Mutex};

use crate::{
    dsp::{time::Clock, AudioRate, ControlRate, Signal},
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
    BlSawOsc,
    EventToAudio,
    MidiToFreq,
    Clock,
    Delay,
    Tape,
}

impl BuiltinNode {
    pub fn create_nodes(
        &self,
        name: &str,
        audio_buffer_len: usize,
    ) -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        match self {
            Self::Sine => crate::dsp::basic::Sine::create_nodes(name, audio_buffer_len, 0.0),
            Self::SineOsc => crate::dsp::generators::SineOsc::create_nodes(
                name,
                audio_buffer_len,
                1.0,
                440.0,
                0.0,
            ),
            Self::BlSawOsc => crate::dsp::generators::BlSawOsc::create_nodes(
                name,
                audio_buffer_len,
                Arc::new(Mutex::new(0.0)),
                Arc::new(Mutex::new(1.0)),
                Arc::new(Mutex::new(0.0)),
                1.0,
                440.0,
            ),
            Self::EventToAudio => {
                crate::dsp::basic::EventToAudio::create_nodes(name, audio_buffer_len, 0.0)
            }
            Self::MidiToFreq => {
                crate::dsp::midi::MidiToFreq::create_nodes(name, audio_buffer_len, 0.0)
            }
            Self::Clock => crate::dsp::time::Clock::create_nodes(name, audio_buffer_len, 1.0, 0.5),
            Self::Delay => crate::dsp::time::Delay::create_nodes(
                name,
                audio_buffer_len,
                Arc::new(Mutex::new(vec![0.0; 480000].into())), // TODO: don't hardcode
                Arc::new(Mutex::new(0.0)),
                1.0,
            ),
            Self::Tape => crate::dsp::time::Tape::create_nodes(
                name,
                audio_buffer_len,
                Arc::new(Mutex::new(vec![].into())),
                Arc::new(Mutex::new(0.0)),
                1.0,
            ),
        }
    }
}

pub fn create_statement<'a>() -> impl FnMut(&'a str) -> IResult<&str, ParsedCreate> {
    map(
        tuple((
            tag("create"),
            whitespace1(),
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
        |(_, _, id, _, graph_name, _control_input_defaults, _)| ParsedCreate {
            ident: id.to_owned(),
            rhs: match graph_name {
                "sin" => CreateRhs::BuiltinNode(BuiltinNode::Sine),
                "sineosc" => CreateRhs::BuiltinNode(BuiltinNode::SineOsc),
                "sawosc" => CreateRhs::BuiltinNode(BuiltinNode::BlSawOsc),
                "e2a" => CreateRhs::BuiltinNode(BuiltinNode::EventToAudio),
                "m2f" => CreateRhs::BuiltinNode(BuiltinNode::MidiToFreq),
                "clock" => CreateRhs::BuiltinNode(BuiltinNode::Clock),
                "delay" => CreateRhs::BuiltinNode(BuiltinNode::Delay),
                "tape" => CreateRhs::BuiltinNode(BuiltinNode::Tape),
                _ => CreateRhs::ScriptGraph(NodeName::new(graph_name)),
            },
        },
    )
}
