use nom::{
    branch::alt, bytes::complete::*, combinator::*, multi::*, number::complete::float, sequence::*,
    IResult,
};
use std::sync::{Arc, Mutex};

use crate::{
    dsp::{AudioRate, ControlRate, Signal},
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
    SineOsc {
        amp: Signal<ControlRate>,
        freq: Signal<ControlRate>,
        fm_amt: Signal<ControlRate>,
    },
    BlSawOsc {
        amp: Signal<ControlRate>,
        freq: Signal<ControlRate>,
    },
    EventToAudio,
    MidiToFreq,
}

impl BuiltinNode {
    pub fn create_graphs(&self, name: &str) -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        match self {
            Self::Sine => crate::dsp::basic::Sine::create_nodes(name, 0.0),
            Self::SineOsc { amp, freq, fm_amt } => crate::dsp::generators::SineOsc::create_nodes(
                name,
                amp.value(),
                freq.value(),
                fm_amt.value(),
            ),
            Self::BlSawOsc { amp, freq } => crate::dsp::generators::BlSawOsc::create_nodes(
                name,
                Arc::new(Mutex::new(0.0)),
                Arc::new(Mutex::new(1.0)),
                Arc::new(Mutex::new(0.0)),
                amp.value(),
                freq.value(),
            ),
            Self::EventToAudio => crate::dsp::basic::EventToAudio::create_nodes(name, 0.0),
            Self::MidiToFreq => crate::dsp::midi::MidiToFreq::create_nodes(name, 0.0),
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
        |(_, _, id, _, graph_name, control_input_defaults, _)| ParsedLet {
            ident: id.to_owned(),
            graph_name: match graph_name {
                "sin" => LetRhs::BuiltinNode(BuiltinNode::Sine),
                "sineosc" => {
                    let defaults = control_input_defaults.unwrap();
                    LetRhs::BuiltinNode(BuiltinNode::SineOsc {
                        amp: Signal::new_control(defaults[0] as Scalar),
                        freq: Signal::new_control(defaults[1] as Scalar),
                        fm_amt: Signal::new_control(defaults[2] as Scalar),
                    })
                }
                "sawosc" => {
                    let defaults = control_input_defaults.unwrap();
                    LetRhs::BuiltinNode(BuiltinNode::BlSawOsc {
                        amp: Signal::new_control(defaults[0] as Scalar),
                        freq: Signal::new_control(defaults[1] as Scalar),
                    })
                }
                "e2a" => LetRhs::BuiltinNode(BuiltinNode::EventToAudio),
                "m2f" => LetRhs::BuiltinNode(BuiltinNode::MidiToFreq),
                _ => LetRhs::ScriptGraph(NodeName(graph_name.to_owned())),
            },
        },
    )
}
