use std::sync::Arc;

use crate::{
    dsp::{
        basic::{UiInputWidget, UiOutputWidget},
        generators::BL_SQUARE_MAX_COEFF,
    },
    graph::Node,
};

use super::{ParsedCreationArg, ParsedIdent};

#[derive(Debug, Clone, Copy)]
pub enum BuiltinNode {
    Sine,
    Cosine,
    Tanh,
    Exp,
    Abs,
    FmSineOsc,
    SineOsc,
    BlSawOsc,
    BlSquareOsc,
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

    // UI nodes
    Slider,
    Button,
    Toggle,
    Led,
}

impl BuiltinNode {
    pub fn try_from_ident(id: &ParsedIdent) -> Option<BuiltinNode> {
        match id.0.as_str() {
            "Sin" => Some(BuiltinNode::Sine),
            "Cos" => Some(BuiltinNode::Cosine),
            "Exp" => Some(BuiltinNode::Exp),
            "Tanh" => Some(BuiltinNode::Tanh),
            "Abs" => Some(BuiltinNode::Abs),
            "SineFm" => Some(BuiltinNode::FmSineOsc),
            "SineOsc" => Some(BuiltinNode::SineOsc),
            "SawOsc" => Some(BuiltinNode::BlSawOsc),
            "SquareOsc" => Some(BuiltinNode::BlSquareOsc),
            "M2F" => Some(BuiltinNode::MidiToFreq),
            "Clock" => Some(BuiltinNode::Clock),
            "Delay" => Some(BuiltinNode::Delay),
            "NoteIn" => Some(BuiltinNode::NoteIn),
            "Redge" => Some(BuiltinNode::RisingEdge),
            "Fedge" => Some(BuiltinNode::FallingEdge),
            "Var" => Some(BuiltinNode::Var),
            "Max" => Some(BuiltinNode::Max),
            "Min" => Some(BuiltinNode::Min),
            "Clip" => Some(BuiltinNode::Clip),
            "Debug" => Some(BuiltinNode::Debug),
            "If" => Some(BuiltinNode::If),
            "Not" => Some(BuiltinNode::Not),
            "Sample" => Some(BuiltinNode::Sample),

            // UI nodes
            "Slider" => Some(BuiltinNode::Slider),
            "Button" => Some(BuiltinNode::Button),
            "Toggle" => Some(BuiltinNode::Toggle),
            "Led" => Some(BuiltinNode::Led),
            _ => None,
        }
    }

    pub fn create_node(&self, name: &str, creation_args: &[ParsedCreationArg]) -> Arc<Node> {
        match self {
            Self::Debug => crate::dsp::basic::DebugNode::create_node(name),
            Self::Sine => crate::dsp::basic::Sine::create_node(name, 0.0),
            Self::Cosine => crate::dsp::basic::Cosine::create_node(name, 0.0),
            Self::Exp => crate::dsp::basic::Exp::create_node(name, 0.0),
            Self::Tanh => crate::dsp::basic::Tanh::create_node(name, 0.0),
            Self::Abs => crate::dsp::basic::Abs::create_node(name, 0.0),
            Self::Not => crate::dsp::basic::Not::create_node(name, 0.0),
            Self::FmSineOsc => {
                crate::dsp::generators::FmSineOsc::create_node(name, 1.0, 440.0, 0.0, 0.0)
            }
            Self::SineOsc => crate::dsp::generators::SineOsc::create_node(name, 1.0, 1.0),
            Self::BlSawOsc => {
                crate::dsp::generators::BlSawOsc::create_node(name, 0.0, 1.0, 0.0, 1.0, 440.0)
            }
            Self::BlSquareOsc => crate::dsp::generators::BlSquareOsc::create_node(
                name,
                [0.0; BL_SQUARE_MAX_COEFF],
                0.0,
                440.0,
                0.5,
            ),
            Self::MidiToFreq => crate::dsp::midi::MidiToFreq::create_node(name, 0.0),
            Self::Clock => crate::dsp::time::Clock::create_node(name, 1.0, 0.5),
            Self::Delay => crate::dsp::time::Delay::create_node(
                name,
                vec![0.0; 480000], // TODO: don't hardcode
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ),
            Self::NoteIn => crate::io::midi::NoteIn::create_node(name),
            Self::RisingEdge => crate::dsp::basic::RisingEdge::create_node(name, 0.0, 0.0),
            Self::FallingEdge => crate::dsp::basic::FallingEdge::create_node(name, 0.0, 0.0),
            Self::Var => crate::dsp::graph_util::Var::create_node(name, 0.0, 0.0, 0.0),
            Self::Max => crate::dsp::basic::Max::create_node(name, 0.0, 0.0),
            Self::Min => crate::dsp::basic::Min::create_node(name, 0.0, 0.0),
            Self::Clip => crate::dsp::basic::Clip::create_node(name, 0.0, 0.0, 0.0),
            Self::If => crate::dsp::basic::If::create_node(name, 0.0, 0.0, 0.0),
            Self::Sample => crate::dsp::samplers::Sample::create_node(
                name,
                creation_args[0].unwrap_string().into(),
            ),

            // UI nodes
            Self::Slider => crate::dsp::basic::UiInput::create_node(
                name,
                *creation_args[2].unwrap_scalar(),
                UiInputWidget::Slider {
                    minimum: *creation_args[0].unwrap_scalar(),
                    maximum: *creation_args[1].unwrap_scalar(),
                },
            ),

            Self::Button => crate::dsp::basic::UiInput::create_node(
                name,
                *creation_args[0].unwrap_scalar(),
                UiInputWidget::Button,
            ),
            Self::Toggle => crate::dsp::basic::UiInput::create_node(
                name,
                *creation_args[0].unwrap_scalar(),
                UiInputWidget::Toggle,
            ),

            Self::Led => crate::dsp::basic::UiOutput::create_node(
                name,
                UiOutputWidget::Led {
                    value: *creation_args[0].unwrap_scalar(),
                },
            ),
        }
    }
}