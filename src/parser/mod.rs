use std::sync::Arc;

use nom::{
    branch::*, bytes::complete::*, character::complete::*, combinator::*, multi::*,
    number::complete::float, sequence::*, IResult,
};
use rustc_hash::FxHashMap;

use crate::{
    dsp::{basic::Constant, Signal},
    graph::{
        AudioRate, Connection, ControlRate, Graph, Input, InputName, Node, NodeName, Output,
        OutputName,
    },
    Scalar,
};

#[derive(Clone)]
pub enum Binding {
    AudioIo(String),
    AudioConstant(Arc<Node<AudioRate>>),
    ControlIo(String),
    ControlConstant(Arc<Node<ControlRate>>),
}

impl Binding {
    pub fn into_input_name(self) -> Option<InputName> {
        match self {
            Self::AudioIo(n) | Self::ControlIo(n) => Some(InputName(n)),
            Self::AudioConstant(_) | Self::ControlConstant(_) => None,
        }
    }

    pub fn into_output_name(self) -> Option<OutputName> {
        match self {
            Self::AudioIo(n) | Self::ControlIo(n) => Some(OutputName(n)),
            Self::AudioConstant(_) | Self::ControlConstant(_) => None,
        }
    }
}

#[derive(Clone)]
pub enum ConnectionOrLet {
    Connection(ParsedConnection),
    Let(ParsedLet),
}

#[derive(Debug, Clone)]
pub struct ParsedLet {
    ident: String,
    graph_name: NodeName,
}

#[derive(Clone)]
pub struct ParsedConnection {
    from_node: Option<NodeName>,
    from_output: Binding,
    to_node: Option<NodeName>,
    to_input: Binding,
}

pub fn whitespace0<'a>() -> impl FnMut(&'a str) -> IResult<&str, &str> {
    recognize(many0(alt((space1, tag("\n"), tag("\r")))))
}

pub fn whitespace1<'a>() -> impl FnMut(&'a str) -> IResult<&str, &str> {
    recognize(many1(alt((space1, tag("\n"), tag("\r")))))
}

pub fn comment<'a>() -> impl FnMut(&'a str) -> IResult<&str, &str> {
    map(delimited(tag("/*"), take_until("*/"), tag("*/")), |_| "")
}

pub fn ignore_garbage<'a, O>(
    parser: impl FnMut(&'a str) -> IResult<&str, O>,
) -> impl FnMut(&'a str) -> IResult<&str, O> {
    map(
        tuple((
            opt(comment()),
            whitespace0(),
            parser,
            whitespace0(),
            opt(comment()),
        )),
        |(_, _, a, _, _)| a,
    )
}

pub fn ident<'a>() -> impl FnMut(&'a str) -> IResult<&str, &str> {
    recognize(pair(alpha1, opt(many1(alt((alphanumeric1, tag("_")))))))
}

pub fn audio_binding<'a>() -> impl FnMut(&'a str) -> IResult<&'a str, Binding> {
    // nom::error::context(
    //     "audio_binding",
    alt((
        map(preceded(tag("@"), ident()), |id| {
            Binding::AudioIo(id.to_owned())
        }),
        map(preceded(tag("@"), float), |num| {
            let (an, cn) = Constant::create_nodes(num as Scalar);
            Binding::AudioConstant(an)
        }),
    ))

    // )
}

pub fn control_binding<'a>() -> impl FnMut(&'a str) -> IResult<&'a str, Binding> {
    // nom::error::context(
    //     "control_binding",
    alt((
        map(preceded(tag("#"), ident()), |id| {
            Binding::ControlIo(id.to_owned())
        }),
        map(preceded(tag("#"), float), |num| {
            let (an, cn) = Constant::create_nodes(num as Scalar);
            Binding::ControlConstant(cn)
        }),
    ))

    // )
}

pub fn in_braces<'a, O>(
    parser: impl FnMut(&'a str) -> IResult<&str, O>,
) -> impl FnMut(&'a str) -> IResult<&str, O> {
    // nom::error::context(
    //     "in_braces",
    map(
        tuple((
            preceded(tuple((tag("{"), space0)), parser),
            recognize(tuple((space0, tag("}")))),
        )),
        |(a, _)| a,
    )
    // )
}

pub fn many_in_braces<'a, O>(
    parser: impl FnMut(&'a str) -> IResult<&str, O>,
) -> impl FnMut(&'a str) -> IResult<&str, Vec<O>> {
    // nom::error::context(
    // "many_in_braces",
    in_braces(preceded(
        whitespace0(),
        many0(map(tuple((parser, whitespace0())), |(p, _)| p)),
    ))
    // )
}

pub fn connection<'a>() -> impl FnMut(&'a str) -> IResult<&str, ParsedConnection> {
    map(
        tuple((
            ignore_garbage(tuple((
                opt(map(tuple((ident(), tag("."))), |(a, _)| a)),
                alt((audio_binding(), control_binding())),
            ))),
            tag("->"),
            ignore_garbage(tuple((
                opt(map(tuple((ident(), tag("."))), |(a, _)| a)),
                alt((audio_binding(), control_binding())),
            ))),
            tag(";"),
        )),
        |((from_node, from_output), _, (to_node, to_input), _)| ParsedConnection {
            from_node: from_node.map(|n| NodeName(n.to_owned())),
            from_output,
            to_node: to_node.map(|n| NodeName(n.to_owned())),
            to_input,
        },
    )
}

pub fn let_statement<'a>() -> impl FnMut(&'a str) -> IResult<&str, ParsedLet> {
    map(
        tuple((
            tag("let"),
            whitespace1(),
            ident(),
            ignore_garbage(tag(":")),
            ident(),
            ignore_garbage(tag(";")),
        )),
        |(_, _, id, _, graph_name, _)| ParsedLet {
            ident: id.to_owned(),
            graph_name: NodeName(graph_name.to_owned()),
        },
    )
}

pub fn connection_or_let(inp: &str) -> IResult<&str, ConnectionOrLet> {
    alt((
        map(connection(), ConnectionOrLet::Connection),
        map(let_statement(), ConnectionOrLet::Let),
    ))(inp)
}

// #[derive(Clone)]
pub struct GraphPtrs {
    pub name: NodeName,
    pub audio: Graph<AudioRate>,
    pub control: Graph<ControlRate>,
}

pub struct ParserContext {
    pub known_node_defs: FxHashMap<NodeName, String>,
}

pub struct ParserScope {
    pub defined_node_instances: FxHashMap<NodeName, GraphPtrs>,
}

pub fn graph_def<'a>() -> impl FnMut(
    &'a str,
) -> IResult<
    &str,
    (
        &str,
        (
            &str,
            Vec<Binding>,
            &str,
            Vec<Binding>,
            &str,
            Vec<Binding>,
            &str,
            Vec<Binding>,
            &str,
            Vec<ConnectionOrLet>,
        ),
    ),
> {
    tuple((
        preceded(ignore_garbage(tag("graph")), ident()),
        ignore_garbage(in_braces(ignore_garbage(tuple((
            ignore_garbage(tag("@in")),
            many_in_braces(ignore_garbage(audio_binding())),
            ignore_garbage(tag("@out")),
            many_in_braces(ignore_garbage(audio_binding())),
            ignore_garbage(tag("#in")),
            many_in_braces(ignore_garbage(control_binding())),
            ignore_garbage(tag("#out")),
            many_in_braces(ignore_garbage(control_binding())),
            ignore_garbage(tag("~")),
            many_in_braces(ignore_garbage(connection_or_let)),
        ))))),
    ))
}

// pub fn parse_let<'a>(inp: &'a str)

pub fn graph_def_instantiation<'a>(
    inp: &'a str,
    ctx: &mut ParserContext,
    _scope: &mut ParserScope,
) -> IResult<&'a str, GraphPtrs> {
    map(
        graph_def(),
        |(
            id,
            (
                _,
                audio_inputs,
                _,
                audio_outputs,
                _,
                control_inputs,
                _,
                control_outputs,
                _,
                connections,
            ),
        )| {
            let id = NodeName(id.to_owned());
            let audio_inputs = audio_inputs
                .into_iter()
                .map(|inp| {
                    (
                        inp.clone().into_input_name().unwrap(),
                        Input::new(&inp.into_input_name().unwrap().0, Signal::new_audio(0.0)),
                    )
                })
                .collect::<FxHashMap<_, _>>();
            let control_inputs = control_inputs
                .into_iter()
                .map(|inp| {
                    (
                        inp.clone().into_input_name().unwrap(),
                        Input::new(&inp.into_input_name().unwrap().0, Signal::new_control(0.0)),
                    )
                })
                .collect::<FxHashMap<_, _>>();
            let audio_outputs = audio_outputs
                .into_iter()
                .map(|out| {
                    (
                        out.clone().into_output_name().unwrap(),
                        Output {
                            name: out.into_output_name().unwrap(),
                        },
                    )
                })
                .collect::<FxHashMap<_, _>>();
            let control_outputs = control_outputs
                .into_iter()
                .map(|out| {
                    (
                        out.clone().into_output_name().unwrap(),
                        Output {
                            name: out.into_output_name().unwrap(),
                        },
                    )
                })
                .collect::<FxHashMap<_, _>>();

            let mut cg = Graph::<ControlRate>::new(control_inputs.clone(), control_outputs.clone());
            let mut ag = Graph::<AudioRate>::new(audio_inputs.clone(), audio_outputs.clone());

            for conn in connections {
                match conn {
                    ConnectionOrLet::Connection(conn) => {
                        macro_rules! io_io_bindings_impl {
                            ($other:ident, $mine:ident, $frm:expr, $to:expr) => {
                                let (source_output, source) =
                                    if let Some(from_node) = conn.from_node {
                                        // cross-graph connection, use the external names for things
                                        (
                                            OutputName($frm.clone()),
                                            $mine.node_id_by_name(&from_node).unwrap(),
                                        )
                                    } else {
                                        // self-output is the source, use our own names for things
                                        (
                                            OutputName("out".to_owned()),
                                            $mine.node_id_by_name(&NodeName($frm.clone())).unwrap(),
                                        )
                                    };
                                let (sink_input, sink) = if let Some(to_node) = conn.to_node {
                                    // cross-graph connection, use the external names for things
                                    (
                                        InputName($to.clone()),
                                        $mine.node_id_by_name(&to_node).unwrap(),
                                    )
                                } else {
                                    // self-input is the sink, use our own names for things
                                    (
                                        InputName("in".to_owned()),
                                        $mine.node_id_by_name(&NodeName($to.clone())).unwrap(),
                                    )
                                };
                                $mine.add_edge(
                                    source,
                                    sink,
                                    Connection {
                                        source_output,
                                        sink_input,
                                    },
                                );
                            };
                        }
                        match (conn.from_output, conn.to_input) {
                            (Binding::AudioIo(frm), Binding::AudioIo(to)) => {
                                io_io_bindings_impl!(audio, ag, frm, to);
                            }
                            (Binding::ControlIo(frm), Binding::ControlIo(to)) => {
                                io_io_bindings_impl!(control, cg, frm, to);
                            }
                            (Binding::AudioConstant(con), Binding::AudioIo(to)) => {
                                todo!("audio constant -> audio io");
                            }
                            (Binding::ControlConstant(con), Binding::ControlIo(to)) => {
                                let (sink_input, sink) = if let Some(to_node) = conn.to_node {
                                    // cross-graph connection, use the external names for things
                                    (
                                        InputName(to.clone()),
                                        cg.node_id_by_name(&to_node).unwrap(),
                                    )
                                } else {
                                    // self-input is the sink, use our own names for things
                                    (
                                        InputName("in".to_owned()),
                                        cg.node_id_by_name(&NodeName(to.clone())).unwrap(),
                                    )
                                };
                                let con_id = cg.add_node(con, Default::default());
                                cg.add_edge(con_id, sink, Connection { source_output: OutputName("out".to_owned()), sink_input });
                            }
                            (Binding::ControlConstant(_), Binding::AudioIo(io)) => panic!("Parsing error: cannot attach control constant to audio input `{io}`"),
                            (Binding::AudioConstant(_), Binding::ControlIo(io)) => panic!("Parsing error: cannot attach audio constant to control input `{io}`"),
                            (_, Binding::AudioConstant(_) | Binding::ControlConstant(_)) => panic!("Parsing error: constant cannot take inputs"),
                            (Binding::AudioIo(frm), Binding::ControlIo(to)) => panic!(
                                "Parsing error: cannot attach audio output `{frm}` to control input `{to}`"
                            ),
                            (Binding::ControlIo(frm), Binding::AudioIo(to)) => panic!(
                                "Parsing error: cannot attach control output `{frm}` to audio input `{to}`"
                            ),
                        }
                    }
                    ConnectionOrLet::Let(pl) => {
                        let mut new_scope = ParserScope {
                            defined_node_instances: FxHashMap::default(),
                        };
                        let (_, graphs) = graph_def_instantiation(
                            &ctx.known_node_defs[&pl.graph_name].clone(),
                            ctx,
                            &mut new_scope,
                        )
                        .unwrap();
                        let GraphPtrs {
                            name: _,
                            audio,
                            control,
                        } = graphs;
                        ag.add_node(Arc::new(Node::from_graph(audio)), &pl.ident);
                        cg.add_node(Arc::new(Node::from_graph(control)), &pl.ident);
                    }
                }
            }

            GraphPtrs {
                name: id.to_owned(),
                audio: ag,
                control: cg,
            }
        },
    )(inp)
}

pub fn parse_script(inp: &str) -> FxHashMap<NodeName, GraphPtrs> {
    let (garbage, defs) = many1(ignore_garbage(recognize(graph_def())))(inp).unwrap();
    if !garbage.is_empty() {
        panic!("Parsing error: couldn't recognize `{garbage}` as graph definition");
    }
    let mut out = FxHashMap::default();
    let mut ctx = ParserContext {
        known_node_defs: FxHashMap::default(),
    };

    for def in defs {
        let (_, (id, _)) = ignore_garbage(graph_def())(def).unwrap();
        ctx.known_node_defs
            .insert(NodeName(id.to_owned()), def.to_owned());
    }

    let main_def = ctx
        .known_node_defs
        .get(&NodeName("main".to_owned()))
        .expect("Parsing error: no `main` graph found")
        .to_owned();

    let mut main_scope = ParserScope {
        defined_node_instances: FxHashMap::default(),
    };
    let (_, main_ptrs) =
        ignore_garbage(|a| graph_def_instantiation(a, &mut ctx, &mut main_scope))(&main_def)
            .unwrap();
    out.insert(NodeName("main".to_owned()), main_ptrs);
    out.extend(main_scope.defined_node_instances);

    out
}

#[cfg(test)]
mod tests {
    use crate::parser::*;

    #[test]
    fn test_parse() {
        let txt = r#"
graph foo {
    @in {}
    @out {}
    #in { #control_input0 }
    #out { #control_output0 }
    ~ {
        #control_input0 -> #control_output0;
    }
}

graph main {
    @in {}
    @out { @dac0 @dac1 }
    #in { #in0 }
    #out { #out0 }
    ~ {
        let f : foo;
        f.#control_output0 -> #in0;
    }
}
"#;
        let graphs = parse_script(txt);
        dbg!(graphs.keys().collect::<Vec<_>>());
    }
}
