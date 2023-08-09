use nom::{
    bytes::complete::*, combinator::*, multi::*, number::complete::float, sequence::*, IResult,
};
use std::sync::{Arc, Mutex};

use crate::{
    dsp::{AudioRate, ControlRate, Signal},
    graph::{Node, NodeName},
    Scalar,
};

use super::{ident, ignore_garbage, whitespace1, LetRhs, ParsedLet};

pub enum BuiltinNode {
    SineOsc {
        amp: Signal<ControlRate>,
        freq: Signal<ControlRate>,
        fm_amt: Signal<ControlRate>,
    },
    BlSawOsc {
        amp: Signal<ControlRate>,
        freq: Signal<ControlRate>,
    },
    MidiToFreq,
}

impl BuiltinNode {
    pub fn create_graphs(&self, name: &str) -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        match self {
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
                "m2f" => LetRhs::BuiltinNode(BuiltinNode::MidiToFreq),
                _ => LetRhs::ScriptGraph(NodeName(graph_name.to_owned())),
            },
        },
    )
}
