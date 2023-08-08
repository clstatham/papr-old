use std::sync::Arc;

use nom::{
    branch::*, bytes::complete::*, character::complete::*, combinator::*, multi::*,
    number::complete::float, sequence::*, IResult,
};
use petgraph::stable_graph::NodeIndex;
use rustc_hash::FxHashMap;

use crate::{
    dsp::{
        basic::{Add, Constant, Divide, Multiply, Subtract},
        AudioRate, ControlRate, Signal,
    },
    graph::{Connection, Graph, Input, InputName, Node, NodeName, Output, OutputName},
    Scalar,
};

#[derive(Clone)]
pub enum Binding {
    AudioIo { node: Option<String>, port: String },
    AudioConstant(Arc<Node<AudioRate>>),
    ControlIo { node: Option<String>, port: String },
    ControlConstant(Arc<Node<ControlRate>>),
}

impl Binding {
    pub fn into_input_name(self) -> Option<InputName> {
        match self {
            Self::AudioIo { port, .. } | Self::ControlIo { port, .. } => Some(InputName(port)),
            Self::AudioConstant(_) | Self::ControlConstant(_) => None,
        }
    }

    pub fn into_output_name(self) -> Option<OutputName> {
        match self {
            Self::AudioIo { port, .. } | Self::ControlIo { port, .. } => Some(OutputName(port)),
            Self::AudioConstant(_) | Self::ControlConstant(_) => None,
        }
    }

    pub fn node(&self) -> Option<&String> {
        match self {
            Self::AudioIo { node, port: _ } | Self::ControlIo { node, port: _ } => node.as_ref(),
            _ => None,
        }
    }

    pub fn port(&self) -> Option<&String> {
        match self {
            Self::AudioIo { node: _, port } | Self::ControlIo { node: _, port } => Some(port),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

pub enum Expr {
    Binding(Binding),
    Expr(Box<Expr>, BinaryOp, Box<Expr>),
}

pub enum ConnectionOrLet {
    Connection(ParsedConnection),
    Let(ParsedLet),
}

#[derive(Debug, Clone)]
pub struct ParsedLet {
    ident: String,
    graph_name: NodeName,
}

pub struct ParsedConnection {
    from: Expr,
    to: Vec<Binding>,
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
        map(
            tuple((
                opt(map(tuple((ident(), tag("."))), |(a, _)| a)),
                preceded(tag("@"), ident()),
            )),
            |(node, port)| Binding::AudioIo {
                node: node.map(ToOwned::to_owned),
                port: port.to_owned(),
            },
        ),
        map(preceded(tag("@"), float), |num| {
            let (an, _cn) = Constant::create_nodes("constant", num as Scalar);
            Binding::AudioConstant(an)
        }),
    ))

    // )
}

pub fn control_binding<'a>() -> impl FnMut(&'a str) -> IResult<&'a str, Binding> {
    // nom::error::context(
    //     "control_binding",
    alt((
        map(
            tuple((
                opt(map(tuple((ident(), tag("."))), |(a, _)| a)),
                preceded(tag("#"), ident()),
            )),
            |(node, port)| Binding::ControlIo {
                node: node.map(ToOwned::to_owned),
                port: port.to_owned(),
            },
        ),
        map(preceded(tag("#"), float), |num| {
            let (_an, cn) = Constant::create_nodes("constant", num as Scalar);
            Binding::ControlConstant(cn)
        }),
    ))

    // )
}

fn solve_expr(
    super_audio: &mut Graph<AudioRate>,
    super_control: &mut Graph<ControlRate>,
    a: &Expr,
    op: BinaryOp,
    b: &Expr,
) -> (bool, NodeIndex) {
    let mut expr_to_node_idxs = |a: &Expr| match a {
        Expr::Binding(a) => match a {
            Binding::AudioConstant(a) => {
                (true, super_audio.add_node(a.clone()), OutputName::default())
            }
            Binding::ControlConstant(a) => (
                false,
                super_control.add_node(a.clone()),
                OutputName::default(),
            ),

            Binding::AudioIo { node, port } => {
                if let Some(node) = node {
                    (
                        true,
                        super_audio
                            .node_id_by_name(&NodeName(node.to_owned()))
                            .unwrap(),
                        OutputName(port.to_owned()),
                    )
                } else {
                    (
                        true,
                        super_audio
                            .node_id_by_name(&NodeName(port.to_owned()))
                            .unwrap(),
                        OutputName::default(),
                    )
                }
            }
            Binding::ControlIo { node, port } => {
                if let Some(node) = node {
                    (
                        false,
                        super_control
                            .node_id_by_name(&NodeName(node.to_owned()))
                            .unwrap(),
                        OutputName(port.to_owned()),
                    )
                } else {
                    (
                        false,
                        super_control
                            .node_id_by_name(&NodeName(port.to_owned()))
                            .unwrap(),
                        OutputName::default(),
                    )
                }
            }
        },
        Expr::Expr(a1, a_op, a2) => {
            let (subexpr_is_audio, subexpr_idx_supergraph) =
                solve_expr(super_audio, super_control, a1, *a_op, a2);
            (
                subexpr_is_audio,
                subexpr_idx_supergraph,
                OutputName::default(),
            )
        }
    };

    let (a_is_audio, a_idx_supergraph, a_out_supergraph) = expr_to_node_idxs(a);
    let (b_is_audio, b_idx_supergraph, b_out_supergraph) = expr_to_node_idxs(b);

    let (mut expr_ag, mut expr_cg) = dual_graphs! {
        "expr"
        @in { "a" = 0.0 "b" = 0.0 }
        @out { "out" }
        #in { "a" = 0.0 "b" = 0.0 }
        #out { "out" }
    };
    let (op_an, op_cn) = match op {
        BinaryOp::Add => Add::create_nodes("add"),
        BinaryOp::Sub => Subtract::create_nodes("sub"),
        BinaryOp::Mul => Multiply::create_nodes("mul"),
        BinaryOp::Div => Divide::create_nodes("div"),
    };
    let op_an = expr_ag.add_node(op_an);
    let op_cn = expr_cg.add_node(op_cn);

    expr_ag.add_edge(
        expr_ag.node_id_by_name(&NodeName("a".to_owned())).unwrap(),
        op_an,
        Connection {
            source_output: OutputName::default(),
            sink_input: InputName("a".to_owned()),
        },
    );
    expr_ag.add_edge(
        expr_ag.node_id_by_name(&NodeName("b".to_owned())).unwrap(),
        op_an,
        Connection {
            source_output: OutputName::default(),
            sink_input: InputName("b".to_owned()),
        },
    );
    expr_ag.add_edge(
        op_an,
        expr_ag
            .node_id_by_name(&NodeName("out".to_owned()))
            .unwrap(),
        Connection {
            source_output: OutputName::default(),
            sink_input: InputName("in".to_owned()),
        },
    );

    expr_cg.add_edge(
        expr_cg.node_id_by_name(&NodeName("a".to_owned())).unwrap(),
        op_cn,
        Connection {
            source_output: OutputName::default(),
            sink_input: InputName("a".to_owned()),
        },
    );
    expr_cg.add_edge(
        expr_cg.node_id_by_name(&NodeName("b".to_owned())).unwrap(),
        op_cn,
        Connection {
            source_output: OutputName::default(),
            sink_input: InputName("b".to_owned()),
        },
    );
    expr_cg.add_edge(
        op_cn,
        expr_cg
            .node_id_by_name(&NodeName("out".to_owned()))
            .unwrap(),
        Connection {
            source_output: OutputName::default(),
            sink_input: InputName("in".to_owned()),
        },
    );

    if a_is_audio && b_is_audio {
        // package it up and insert into supergraph
        let expr_an = Arc::new(expr_ag.into_node());
        let expr_an_idx_supergraph = super_audio.add_node(expr_an);
        super_audio.add_edge(
            a_idx_supergraph,
            expr_an_idx_supergraph,
            Connection {
                source_output: a_out_supergraph,
                sink_input: InputName("a".to_owned()),
            },
        );
        super_audio.add_edge(
            b_idx_supergraph,
            expr_an_idx_supergraph,
            Connection {
                source_output: b_out_supergraph,
                sink_input: InputName("b".to_owned()),
            },
        );
        (true, expr_an_idx_supergraph)
    } else if !a_is_audio && !b_is_audio {
        let expr_cn = Arc::new(expr_cg.into_node());
        let expr_cn_idx_supergraph = super_control.add_node(expr_cn);
        super_control.add_edge(
            a_idx_supergraph,
            expr_cn_idx_supergraph,
            Connection {
                source_output: a_out_supergraph,
                sink_input: InputName("a".to_owned()),
            },
        );
        super_control.add_edge(
            b_idx_supergraph,
            expr_cn_idx_supergraph,
            Connection {
                source_output: b_out_supergraph,
                sink_input: InputName("b".to_owned()),
            },
        );
        (false, expr_cn_idx_supergraph)
    } else {
        panic!("Parsing error: mixing audio/control signals in expr")
    }
}

pub fn expr(inp: &str) -> IResult<&str, Expr> {
    alt((
        map(alt((audio_binding(), control_binding())), |b| {
            Expr::Binding(b)
        }),
        map(
            delimited(
                ignore_garbage(tag("(")),
                tuple((
                    ignore_garbage(expr),
                    ignore_garbage(alt((
                        value(BinaryOp::Add, tag("+")),
                        value(BinaryOp::Sub, tag("-")),
                        value(BinaryOp::Mul, tag("*")),
                        value(BinaryOp::Div, tag("/")),
                    ))),
                    ignore_garbage(expr),
                )),
                ignore_garbage(tag(")")),
            ),
            |(a, op, b)| Expr::Expr(Box::new(a), op, Box::new(b)),
        ),
    ))(inp)
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
            ignore_garbage(alt((
                map(alt((audio_binding(), control_binding())), |b| vec![b]),
                delimited(
                    ignore_garbage(tag("[")),
                    many1(ignore_garbage(alt((audio_binding(), control_binding())))),
                    ignore_garbage(tag("]")),
                ),
            ))),
            tag("<-"),
            ignore_garbage(expr),
            tag(";"),
        )),
        |(to_inputs, _, xpr, _)| ParsedConnection {
            from: xpr,
            to: to_inputs,
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

pub struct GraphPtrs {
    pub name: NodeName,
    pub audio: Graph<AudioRate>,
    pub control: Graph<ControlRate>,
}

pub struct ParserContext {
    pub known_node_defs: FxHashMap<NodeName, String>,
}

pub fn graph_def<'a>() -> impl FnMut(
    &'a str,
) -> IResult<
    &str,
    (
        &str,
        (
            Vec<Binding>,
            Vec<Binding>,
            Vec<Binding>,
            Vec<Binding>,
            Vec<ConnectionOrLet>,
        ),
    ),
> {
    tuple((
        preceded(ignore_garbage(tag("graph")), ident()),
        ignore_garbage(in_braces(ignore_garbage(tuple((
            preceded(
                ignore_garbage(tag("@in")),
                many_in_braces(ignore_garbage(audio_binding())),
            ),
            preceded(
                ignore_garbage(tag("@out")),
                many_in_braces(ignore_garbage(audio_binding())),
            ),
            preceded(
                ignore_garbage(tag("#in")),
                many_in_braces(ignore_garbage(control_binding())),
            ),
            preceded(
                ignore_garbage(tag("#out")),
                many_in_braces(ignore_garbage(control_binding())),
            ),
            preceded(
                ignore_garbage(tag("~")),
                many_in_braces(ignore_garbage(connection_or_let)),
            ),
        ))))),
    ))
}

pub fn graph_def_instantiation<'a>(
    inp: &'a str,
    ctx: &mut ParserContext,
) -> IResult<&'a str, GraphPtrs> {
    map(
        graph_def(),
        |(id, (audio_inputs, audio_outputs, control_inputs, control_outputs, connections))| {
            let id = NodeName(id.to_owned());
            let audio_inputs = audio_inputs
                .into_iter()
                .map(|inp| Input::new(&inp.into_input_name().unwrap().0, Signal::new_audio(0.0)))
                .collect::<Vec<_>>();
            let control_inputs = control_inputs
                .into_iter()
                .map(|inp| Input::new(&inp.into_input_name().unwrap().0, Signal::new_control(0.0)))
                .collect::<Vec<_>>();
            let audio_outputs = audio_outputs
                .into_iter()
                .map(|out| Output {
                    name: out.into_output_name().unwrap(),
                })
                .collect::<Vec<_>>();
            let control_outputs = control_outputs
                .into_iter()
                .map(|out| Output {
                    name: out.into_output_name().unwrap(),
                })
                .collect::<Vec<_>>();

            let mut cg = Graph::<ControlRate>::new(
                Some(id.to_owned()),
                control_inputs.clone(),
                control_outputs.clone(),
            );
            let mut ag = Graph::<AudioRate>::new(
                Some(id.to_owned()),
                audio_inputs.clone(),
                audio_outputs.clone(),
            );

            for conn in connections {
                match conn {
                    ConnectionOrLet::Connection(conn) => {
                        let (from_is_audio, from, from_output) = match &conn.from {
                            Expr::Binding(from) => match from {
                                Binding::AudioIo { node, port } => {
                                    if let Some(node) = node {
                                        (
                                            true,
                                            ag.node_id_by_name(&NodeName(node.to_owned())).unwrap(),
                                            OutputName(port.clone()),
                                        )
                                    } else {
                                        (
                                            true,
                                            ag.node_id_by_name(&NodeName(port.to_owned())).unwrap(),
                                            OutputName::default(),
                                        )
                                    }
                                }
                                Binding::ControlIo { node, port } => {
                                    if let Some(node) = node {
                                        (
                                            false,
                                            cg.node_id_by_name(&NodeName(node.to_owned())).unwrap(),
                                            OutputName(port.clone()),
                                        )
                                    } else {
                                        (
                                            false,
                                            cg.node_id_by_name(&NodeName(port.to_owned())).unwrap(),
                                            OutputName::default(),
                                        )
                                    }
                                }
                                Binding::AudioConstant(con) => {
                                    let idx = ag.add_node(con.to_owned());
                                    (true, idx, OutputName::default())
                                }
                                Binding::ControlConstant(con) => {
                                    let idx = cg.add_node(con.to_owned());
                                    (false, idx, OutputName::default())
                                }
                            },
                            Expr::Expr(a, op, b) => {
                                let (is_audio, idx) = solve_expr(&mut ag, &mut cg, a, *op, b);
                                (is_audio, idx, OutputName::default())
                            }
                        };
                        for conn_to in &conn.to {
                            match (from_is_audio, conn_to) {
                                (true, Binding::AudioIo { port: to, .. }) => {
                                    let (sink_input, sink) = if let Some(to_node) = conn_to.node() {
                                        // cross-graph connection, use the external names for things
                                        (
                                            InputName(to.clone()),
                                            ag.node_id_by_name(&NodeName(to_node.to_owned())).unwrap(),
                                        )
                                    } else {
                                        // self-input is the sink, use our own names for things
                                        (
                                            InputName::default(),
                                            ag.node_id_by_name(&NodeName(to.clone())).unwrap(),
                                        )
                                    };
                                    ag.add_edge(from, sink, Connection { source_output: from_output.clone(), sink_input });
                                }
                                (false, Binding::ControlIo { port: to, .. }) => {
                                    let (sink_input, sink) = if let Some(to_node) = conn_to.node() {
                                        // cross-graph connection, use the external names for things
                                        (
                                            InputName(to.clone()),
                                            cg.node_id_by_name(&NodeName(to_node.to_owned())).unwrap(),
                                        )
                                    } else {
                                        // self-input is the sink, use our own names for things
                                        (
                                            InputName::default(),
                                            cg.node_id_by_name(&NodeName(to.clone())).unwrap(),
                                        )
                                    };
                                    // let con_id = cg.add_node(con.to_owned(), Default::default());
                                    cg.add_edge(from, sink, Connection { source_output: from_output.clone(), sink_input });
                                }
                                (false, Binding::AudioIo { port: io, .. }) => panic!("Parsing error: cannot attach control constant to audio input `{io}`"),
                                (true, Binding::ControlIo { port: io, .. }) => panic!("Parsing error: cannot attach audio constant to control input `{io}`"),
                                (_, Binding::AudioConstant(_) | Binding::ControlConstant(_)) => panic!("Parsing error: constant cannot take inputs"),
                            }
                        }
                    }

                    ConnectionOrLet::Let(pl) => {
                        let (_, graphs) = graph_def_instantiation(
                            &ctx.known_node_defs[&pl.graph_name].clone(),
                            ctx,
                        )
                        .unwrap();
                        let GraphPtrs {
                            name: _,
                            mut audio,
                            mut control,
                        } = graphs;
                        audio.name = NodeName(pl.ident.to_owned());
                        control.name = NodeName(pl.ident.to_owned());
                        ag.add_node(Arc::new(Node::from_graph(audio)));
                        cg.add_node(Arc::new(Node::from_graph(control)));
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

    let (_, main_ptrs) =
        ignore_garbage(|a| graph_def_instantiation(a, &mut ctx))(&main_def).unwrap();
    out.insert(NodeName("main".to_owned()), main_ptrs);

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
