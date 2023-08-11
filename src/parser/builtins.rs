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

use super::{ident, ignore_garbage, whitespace1, LetRhs, ParsedLet};

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
                1.0,
            ),
        }
    }
}

pub fn let_statement<'a>() -> impl FnMut(&'a str) -> IResult<&str, ParsedLet> {
    map(
        tuple((
            tag("let"),
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
        |(_, _, id, _, graph_name, _control_input_defaults, _)| ParsedLet {
            ident: id.to_owned(),
            graph_name: match graph_name {
                "sin" => LetRhs::BuiltinNode(BuiltinNode::Sine),
                "sineosc" => LetRhs::BuiltinNode(BuiltinNode::SineOsc),
                "sawosc" => LetRhs::BuiltinNode(BuiltinNode::BlSawOsc),
                "e2a" => LetRhs::BuiltinNode(BuiltinNode::EventToAudio),
                "m2f" => LetRhs::BuiltinNode(BuiltinNode::MidiToFreq),
                "clock" => LetRhs::BuiltinNode(BuiltinNode::Clock),
                "delay" => LetRhs::BuiltinNode(BuiltinNode::Delay),
                _ => LetRhs::ScriptGraph(NodeName::new(graph_name)),
            },
        },
    )
}
