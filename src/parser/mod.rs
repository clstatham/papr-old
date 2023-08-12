use std::sync::Arc;

use nom::{
    branch::*, bytes::complete::*, character::complete::*, combinator::*, multi::*,
    number::complete::float, sequence::*, IResult,
};
use petgraph::stable_graph::NodeIndex;
use std::collections::BTreeMap;

use crate::{
    dsp::{
        basic::{Add, Constant, Divide, Multiply, Subtract},
        graph_util::LetBinding,
        AudioRate, ControlRate, Signal,
    },
    graph::{Connection, Graph, Input, Node, NodeName, Output},
    Scalar,
};

use self::builtins::{create_statement, global_const, BuiltinNode};

pub mod builtins;

#[derive(Clone)]
pub enum Binding {
    AudioIo {
        node: Option<String>,
        port: String,
    },
    AudioConstant(Arc<Node<AudioRate>>),
    ControlIo {
        node: Option<String>,
        port: String,
        default: Option<Signal<ControlRate>>,
    },
    ControlIoBounded {
        port: String,
        min: Signal<ControlRate>,
        max: Signal<ControlRate>,
        default: Option<Signal<ControlRate>>,
    },
    ControlConstant(Arc<Node<ControlRate>>),
}

impl Binding {
    pub fn into_input_name(self) -> Option<String> {
        match self {
            Self::AudioIo { port, .. } | Self::ControlIo { port, .. } => Some(port),
            Self::ControlIoBounded { port, .. } => Some(port),
            Self::AudioConstant(_) | Self::ControlConstant(_) => None,
        }
    }

    pub fn into_output_name(self) -> Option<String> {
        match self {
            Self::AudioIo { port, .. } | Self::ControlIo { port, .. } => Some(port),
            Self::ControlIoBounded { port, .. } => Some(port),
            Self::AudioConstant(_) | Self::ControlConstant(_) => None,
        }
    }

    pub fn node(&self) -> Option<&String> {
        match self {
            Self::AudioIo { node, .. } | Self::ControlIo { node, .. } => node.as_ref(),
            _ => None,
        }
    }

    pub fn port(&self) -> Option<&String> {
        match self {
            Self::AudioIo { port, .. }
            | Self::ControlIo { port, .. }
            | Self::ControlIoBounded { port, .. } => Some(port),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, derive_more::Display)]
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

impl Expr {
    pub fn name(&self) -> String {
        match self {
            Self::Binding(b) => match b {
                Binding::AudioConstant(_) => "@constant".to_string(),
                Binding::ControlConstant(_) => "#constant".to_string(),
                Binding::AudioIo { node, port } => {
                    if let Some(node) = node {
                        format!("{node}.@{port}")
                    } else {
                        format!("@{port}")
                    }
                }
                Binding::ControlIo {
                    node,
                    port,
                    default: _,
                } => {
                    if let Some(node) = node {
                        format!("{node}.#{port}")
                    } else {
                        format!("#{port}")
                    }
                }
                Binding::ControlIoBounded { port, .. } => {
                    format!("@{port}")
                }
            },
            Self::Expr(a, op, b) => format!("({} {} {})", a.name(), op, b.name()),
        }
    }
}

pub enum Statement {
    Connection(ParsedConnection),
    Create(ParsedCreate),
}

pub enum CreateRhs {
    ScriptGraph(NodeName),
    BuiltinNode(BuiltinNode),
}

pub struct ParsedCreate {
    ident: String,
    rhs: CreateRhs,
}

pub struct ParsedConnection {
    is_let: bool,
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

pub fn global_const_or_const<'a>() -> impl FnMut(&'a str) -> IResult<&str, (String, Scalar)> {
    alt((global_const(), map(float, |f| (f.to_string(), f as Scalar))))
}

pub fn audio_binding<'a>(
    audio_buffer_len: usize,
) -> impl FnMut(&'a str) -> IResult<&'a str, Binding> {
    // nom::error::context(
    //     "audio_binding",
    alt((
        map(
            preceded(tag("@"), global_const_or_const()),
            move |(name, num)| {
                let (an, _cn) = Constant::create_nodes(&name, audio_buffer_len, num as Scalar);
                Binding::AudioConstant(an)
            },
        ),
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
    ))

    // )
}

pub fn control_input_binding<'a>() -> impl FnMut(&'a str) -> IResult<&'a str, Binding> {
    map(
        tuple((
            opt(map(tuple((ident(), tag("."))), |(a, _)| a)),
            preceded(tag("#"), ident()),
            opt(delimited(
                ignore_garbage(tag("(")),
                tuple((
                    ignore_garbage(float),
                    ignore_garbage(tag(":")),
                    ignore_garbage(float),
                )),
                ignore_garbage(tag(")")),
            )),
            opt(preceded(ignore_garbage(tag("=")), ignore_garbage(float))),
        )),
        |(node, port, bounds, default)| {
            if let Some((min, _, max)) = bounds {
                Binding::ControlIoBounded {
                    port: port.to_owned(),
                    min: Signal::new_control(min as Scalar),
                    max: Signal::new_control(max as Scalar),
                    default: default.map(|d| Signal::new_control(d as Scalar)),
                }
            } else {
                Binding::ControlIo {
                    node: node.map(ToOwned::to_owned),
                    port: port.to_owned(),
                    default: default.map(|d| Signal::new_control(d as Scalar)),
                }
            }
        },
    )
}

pub fn control_binding<'a>(
    audio_buffer_len: usize,
) -> impl FnMut(&'a str) -> IResult<&'a str, Binding> {
    // nom::error::context(
    //     "control_binding",
    alt((
        map(
            preceded(tag("#"), global_const_or_const()),
            move |(name, num)| {
                let (_an, cn) = Constant::create_nodes(&name, audio_buffer_len, num as Scalar);
                Binding::ControlConstant(cn)
            },
        ),
        map(
            tuple((
                opt(map(tuple((ident(), tag("."))), |(a, _)| a)),
                preceded(tag("#"), ident()),
            )),
            |(node, port)| Binding::ControlIo {
                node: node.map(ToOwned::to_owned),
                port: port.to_owned(),
                default: None,
            },
        ),
    ))

    // )
}

fn solve_expr(
    super_audio: &mut Graph<AudioRate>,
    super_control: &mut Graph<ControlRate>,
    ctx: &ParserContext,
    a: &Expr,
    op: BinaryOp,
    b: &Expr,
) -> (bool, NodeIndex) {
    let mut expr_to_node_idxs = |a: &Expr| match a {
        Expr::Binding(a) => match a {
            Binding::AudioConstant(a) => (true, super_audio.add_node(a.clone()), 0),
            Binding::ControlConstant(a) => (false, super_control.add_node(a.clone()), 0),

            Binding::AudioIo { node, port } => {
                if let Some(node) = node {
                    (
                        true,
                        super_audio.node_id_by_name(node).unwrap_or_else(|| {
                            panic!("Parsing error: No node named `{node}` found on supergraph")
                        }),
                        super_audio.digraph[super_audio.node_id_by_name(node).unwrap()]
                            .output_named(port)
                            .unwrap(),
                    )
                } else {
                    (
                        true,
                        super_audio.node_id_by_name(port).unwrap_or_else(|| {
                            panic!("Parsing error: No node named `{port}` found on supergraph")
                        }),
                        0,
                    )
                }
            }
            Binding::ControlIoBounded { .. } => {
                unreachable!()
            }
            Binding::ControlIo {
                node,
                port,
                default: _,
            } => {
                if let Some(node) = node {
                    (
                        false,
                        super_control.node_id_by_name(node).unwrap_or_else(|| {
                            panic!("Parsing error: No node named `{node}` found on supergraph")
                        }),
                        super_control.digraph[super_control.node_id_by_name(node).unwrap()]
                            .output_named(port)
                            .unwrap(),
                    )
                } else {
                    (
                        false,
                        super_control.node_id_by_name(port).unwrap_or_else(|| {
                            panic!("Parsing error: No node named `{port}` found on supergraph")
                        }),
                        0,
                    )
                }
            }
        },
        Expr::Expr(a1, a_op, a2) => {
            let (subexpr_is_audio, subexpr_idx_supergraph) =
                solve_expr(super_audio, super_control, ctx, a1, *a_op, a2);
            (subexpr_is_audio, subexpr_idx_supergraph, 0)
        }
    };

    let (a_is_audio, a_idx_supergraph, a_out_supergraph) = expr_to_node_idxs(a);
    let (b_is_audio, b_idx_supergraph, b_out_supergraph) = expr_to_node_idxs(b);

    let (mut expr_ag, mut expr_cg) = dual_graphs! {
        &format!("({} {} {})", a.name(), op, b.name());
        ctx.audio_buffer_len;
        @in { "a" = 0.0 "b" = 0.0 }
        @out { "out" }
        #in { "a" = 0.0 "b" = 0.0 }
        #out { "out" }
    };
    let (op_an, op_cn) = match op {
        BinaryOp::Add => Add::create_nodes("add", ctx.audio_buffer_len, 0.0, 0.0),
        BinaryOp::Sub => Subtract::create_nodes("sub", ctx.audio_buffer_len, 0.0, 0.0),
        BinaryOp::Mul => Multiply::create_nodes("mul", ctx.audio_buffer_len, 0.0, 0.0),
        BinaryOp::Div => Divide::create_nodes("div", ctx.audio_buffer_len, 0.0, 0.0),
    };
    let op_an = expr_ag.add_node(op_an);
    let op_cn = expr_cg.add_node(op_cn);

    expr_ag.add_edge(
        expr_ag.node_id_by_name("a").unwrap(),
        op_an,
        Connection {
            source_output: 0,
            sink_input: 0,
        },
    );
    expr_ag.add_edge(
        expr_ag.node_id_by_name("b").unwrap(),
        op_an,
        Connection {
            source_output: 0,
            sink_input: 1,
        },
    );
    expr_ag.add_edge(
        op_an,
        expr_ag.node_id_by_name("out").unwrap(),
        Connection {
            source_output: 0,
            sink_input: 0,
        },
    );

    expr_cg.add_edge(
        expr_cg.node_id_by_name("a").unwrap(),
        op_cn,
        Connection {
            source_output: 0,
            sink_input: 0,
        },
    );
    expr_cg.add_edge(
        expr_cg.node_id_by_name("b").unwrap(),
        op_cn,
        Connection {
            source_output: 0,
            sink_input: 1,
        },
    );
    expr_cg.add_edge(
        op_cn,
        expr_cg.node_id_by_name("out").unwrap(),
        Connection {
            source_output: 0,
            sink_input: 0,
        },
    );

    if a_is_audio && b_is_audio {
        // package it up and insert into supergraph
        let expr_an = Arc::new(expr_ag.into_node());
        let expr_an_idx_supergraph = super_audio.add_node(expr_an.clone());
        super_audio.add_edge(
            a_idx_supergraph,
            expr_an_idx_supergraph,
            Connection {
                source_output: a_out_supergraph,
                sink_input: 0,
            },
        );
        super_audio.add_edge(
            b_idx_supergraph,
            expr_an_idx_supergraph,
            Connection {
                source_output: b_out_supergraph,
                sink_input: 1,
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
                sink_input: 0,
            },
        );
        super_control.add_edge(
            b_idx_supergraph,
            expr_cn_idx_supergraph,
            Connection {
                source_output: b_out_supergraph,
                sink_input: 1,
            },
        );
        (false, expr_cn_idx_supergraph)
    } else {
        panic!("Parsing error: mixing audio/control signals in expr")
    }
}

pub fn expr(inp: &str, audio_buffer_len: usize) -> IResult<&str, Expr> {
    alt((
        map(
            alt((
                audio_binding(audio_buffer_len),
                control_binding(audio_buffer_len),
            )),
            Expr::Binding,
        ),
        map(
            delimited(
                ignore_garbage(tag("(")),
                tuple((
                    ignore_garbage(|a| expr(a, audio_buffer_len)),
                    ignore_garbage(alt((
                        value(BinaryOp::Add, tag("+")),
                        value(BinaryOp::Sub, tag("-")),
                        value(BinaryOp::Mul, tag("*")),
                        value(BinaryOp::Div, tag("/")),
                    ))),
                    ignore_garbage(|a| expr(a, audio_buffer_len)),
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

pub fn connection<'a>(
    audio_buffer_len: usize,
) -> impl FnMut(&'a str) -> IResult<&str, ParsedConnection> {
    map(
        tuple((
            ignore_garbage(alt((
                map(
                    tuple((
                        map(opt(ignore_garbage(tag("let"))), |l| l.is_some()),
                        alt((
                            audio_binding(audio_buffer_len),
                            control_binding(audio_buffer_len),
                        )),
                    )),
                    |(is_let, b)| (is_let, vec![b]),
                ),
                map(
                    delimited(
                        ignore_garbage(tag("[")),
                        many1(ignore_garbage(alt((
                            audio_binding(audio_buffer_len),
                            control_binding(audio_buffer_len),
                        )))),
                        ignore_garbage(tag("]")),
                    ),
                    |v| (false, v),
                ),
            ))),
            tag("<-"),
            ignore_garbage(move |a| expr(a, audio_buffer_len)),
            tag(";"),
        )),
        |((is_let, to_inputs), _, xpr, _)| ParsedConnection {
            is_let,
            from: xpr,
            to: to_inputs,
        },
    )
}

pub fn statement(inp: &str, audio_buffer_len: usize) -> IResult<&str, Statement> {
    alt((
        map(connection(audio_buffer_len), Statement::Connection),
        map(create_statement(), Statement::Create),
    ))(inp)
}

pub struct DualGraphs {
    pub name: NodeName,
    pub audio: Graph<AudioRate>,
    pub control: Graph<ControlRate>,
}

pub struct ParserContext {
    known_node_defs: BTreeMap<NodeName, String>,
    audio_buffer_len: usize,
}

pub fn graph_def<'a>(
    audio_buffer_len: usize,
) -> impl FnMut(
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
            Vec<Statement>,
        ),
    ),
> {
    tuple((
        preceded(ignore_garbage(tag("graph")), ident()),
        ignore_garbage(in_braces(ignore_garbage(tuple((
            preceded(
                ignore_garbage(tag("@in")),
                many_in_braces(ignore_garbage(audio_binding(audio_buffer_len))),
            ),
            preceded(
                ignore_garbage(tag("@out")),
                many_in_braces(ignore_garbage(audio_binding(audio_buffer_len))),
            ),
            preceded(
                ignore_garbage(tag("#in")),
                many_in_braces(ignore_garbage(control_input_binding())),
            ),
            preceded(
                ignore_garbage(tag("#out")),
                many_in_braces(ignore_garbage(control_binding(audio_buffer_len))),
            ),
            preceded(
                ignore_garbage(tag("~")),
                many_in_braces(ignore_garbage(move |a| statement(a, audio_buffer_len))),
            ),
        ))))),
    ))
}

pub fn graph_def_instantiation<'a>(
    inp: &'a str,
    ctx: &mut ParserContext,
) -> IResult<&'a str, DualGraphs> {
    map(
        graph_def(ctx.audio_buffer_len),
        |(id, (audio_inputs, audio_outputs, control_inputs, control_outputs, connections))| {
            let id = NodeName::new(id);
            let audio_inputs = audio_inputs
                .into_iter()
                .map(|inp| {
                    Input::new(
                        &inp.into_input_name().unwrap().to_string(),
                        Some(Signal::new_audio(0.0)),
                    )
                })
                .collect::<Vec<_>>();
            let control_inputs = control_inputs
                .into_iter()
                .map(|inp| {
                    if let Binding::ControlIoBounded {
                        port,
                        min,
                        max,
                        default,
                    } = inp
                    {
                        Input {
                            name: port.to_owned(),
                            minimum: Some(min),
                            maximum: Some(max),
                            default: Some(default.expect(
                                "Parsing error: no default value provided for control input",
                            )),
                            implicit: false,
                            is_ui: true,
                        }
                    } else if let Binding::ControlIo {
                        node: _,
                        port,
                        default,
                    } = inp
                    {
                        Input::new(
                            &port,
                            Some(default.expect(
                                "Parsing error: no default value provided for control input",
                            )),
                        )
                    } else {
                        unreachable!()
                    }
                })
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
                ctx.audio_buffer_len,
                audio_inputs.clone(),
                audio_outputs.clone(),
            );

            for conn in connections {
                match conn {
                    Statement::Connection(conn) => {
                        let (from_is_audio, from, from_output) = match &conn.from {
                            Expr::Binding(from) => match from {
                                Binding::AudioIo { node, port } => {
                                    if let Some(node) = node {
                                        let node_id = ag.node_id_by_name(node).unwrap();
                                        let port = ag.digraph[node_id].output_named(port).unwrap();
                                        (true, node_id, port)
                                    } else {
                                        let node_id = ag.node_id_by_name(port).unwrap();
                                        (true, node_id, 0)
                                    }
                                }
                                Binding::ControlIoBounded { .. } => unreachable!(),
                                Binding::ControlIo {
                                    node,
                                    port,
                                    default: _,
                                } => {
                                    if let Some(node) = node {
                                        let node_id = cg.node_id_by_name(node).unwrap();
                                        let port = cg.digraph[node_id].output_named(port).unwrap();
                                        (false, node_id, port)
                                    } else {
                                        let node_id = cg.node_id_by_name(port).unwrap();
                                        (false, node_id, 0)
                                    }
                                }
                                Binding::AudioConstant(con) => {
                                    let idx = ag.add_node(con.to_owned());
                                    (true, idx, 0)
                                }
                                Binding::ControlConstant(con) => {
                                    let idx = cg.add_node(con.to_owned());
                                    (false, idx, 0)
                                }
                            },
                            Expr::Expr(a, op, b) => {
                                let (is_audio, idx) = solve_expr(&mut ag, &mut cg, ctx, a, *op, b);
                                (is_audio, idx, 0)
                            }
                        };
                        for conn_to in &conn.to {
                            match (from_is_audio, conn_to) {
                                (true, Binding::AudioIo { port: to, .. }) => {
                                    let (sink_input, sink) = if let Some(to_node) = conn_to.node() {
                                        // cross-graph connection, use the external names for things
                                        assert!(!conn.is_let, "Parsing error: Invalid `let` statement");
                                        let node_id = ag.node_id_by_name(to_node).unwrap();
                                        (
                                            ag.digraph[node_id].input_named(to).unwrap(),
                                            node_id,
                                        )
                                    } else {
                                        // self-input is the sink, use our own names for things
                                        if conn.is_let {
                                            let (an, _cn) = LetBinding::create_nodes(to, ctx.audio_buffer_len, 0.0);
                                            (
                                                0,
                                                ag.add_node(an),
                                            )
                                        } else {
                                            (
                                                0,
                                                ag.node_id_by_name(to).unwrap(),
                                            )
                                        }
                                    };
                                    ag.add_edge(from, sink, Connection { source_output: from_output, sink_input });
                                }
                                (false, Binding::ControlIo { port: to, .. }) => {
                                    let (sink_input, sink) = if let Some(to_node) = conn_to.node() {
                                        // cross-graph connection, use the external names for things
                                        assert!(!conn.is_let, "Parsing error: Invalid `let` statement");
                                        let node_id = cg.node_id_by_name(to_node).unwrap();
                                        (
                                            cg.digraph[node_id].input_named(to).unwrap(),
                                            node_id,
                                        )
                                    } else {
                                        // self-input is the sink, use our own names for things
                                        if conn.is_let {
                                            let (_an, cn) = LetBinding::create_nodes(to, ctx.audio_buffer_len, 0.0);
                                            (0,
                                                cg.add_node(cn),
                                            )
                                        } else {
                                            (
                                                0,
                                                cg.node_id_by_name(to).unwrap(),
                                            )
                                        }
                                        
                                    };
                                    // let con_id = cg.add_node(con.to_owned(), Default::default());
                                    cg.add_edge(from, sink, Connection { source_output: from_output, sink_input });
                                }
                                (false, Binding::AudioIo { port: io, .. }) => panic!("Parsing error: cannot attach control constant to audio input `{io}`"),
                                (true, Binding::ControlIo { port: io, .. }) => panic!("Parsing error: cannot attach audio constant to control input `{io}`"),
                                (_, Binding::AudioConstant(_) | Binding::ControlConstant(_)) => panic!("Parsing error: constant cannot take inputs"),
                                (_, Binding::ControlIoBounded { .. }) => unreachable!(),
                            }
                        }
                    }

                    Statement::Create(pl) => {
                        let known_node_defs = ctx.known_node_defs.clone();
                        let (an, cn) = match &pl.rhs {
                            CreateRhs::ScriptGraph(graph_name) => {
                                let (_, graphs) =
                                    graph_def_instantiation(&known_node_defs[graph_name], ctx)
                                        .unwrap();
                                let DualGraphs {
                                    name: _,
                                    mut audio,
                                    mut control,
                                } = graphs;
                                audio.name = NodeName::new(&pl.ident.to_owned());
                                control.name = NodeName::new(&pl.ident.to_owned());
                                (
                                    Arc::new(Node::from_graph(audio)),
                                    Arc::new(Node::from_graph(control)),
                                )
                            }
                            CreateRhs::BuiltinNode(node) => {
                                node.create_nodes(&pl.ident, ctx.audio_buffer_len)
                            }
                        };

                        ag.add_node(an);
                        cg.add_node(cn);
                    }
                }
            }

            ag.write_dot(&format!("{}_audio.dot", id));
            cg.write_dot(&format!("{}_control.dot", id));

            DualGraphs {
                name: id.to_owned(),
                audio: ag,
                control: cg,
            }
        },
    )(inp)
}

pub fn parse_script(
    inp: &str,
    audio_buffer_len: usize,
) -> BTreeMap<NodeName, DualGraphs> {
    let (garbage, defs) =
        many1(ignore_garbage(recognize(graph_def(audio_buffer_len))))(inp).unwrap();
    if !garbage.is_empty() {
        panic!("Parsing error: couldn't recognize `{garbage}` as graph definition");
    }
    let mut out = BTreeMap::default();
    let mut ctx = ParserContext {
        known_node_defs: BTreeMap::default(),
        audio_buffer_len,
    };

    for def in defs {
        let (_, (id, _)) = ignore_garbage(graph_def(audio_buffer_len))(def).unwrap();
        ctx.known_node_defs
            .insert(NodeName::new(id), def.to_owned());
    }

    let main_def = ctx
        .known_node_defs
        .get(&NodeName::new("main"))
        .expect("Parsing error: no `main` graph found")
        .to_owned();

    let (_, main_ptrs) =
        ignore_garbage(|a| graph_def_instantiation(a, &mut ctx))(&main_def).unwrap();
    out.insert(NodeName::new("main"), main_ptrs);

    out
}
