use std::{
    ops::{Range, RangeFrom, RangeFull, RangeTo},
    sync::Arc,
};

use nom::{
    branch::*, bytes::complete::*, character::complete::*, combinator::*, multi::*,
    number::complete::float, sequence::*, IResult, InputIter, InputLength, InputTake,
};

use crate::{
    dsp::{basic::ControlToAudio, graph_util::LetBinding, Signal, SignalRate},
    graph::{Connection, Graph, Input, NodeName, Output},
    Scalar,
};

pub mod builtins;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Whitespace,
    Number(Scalar),
    Ident(String),
    Graph,
    Var,
    Let,
    Do,
    Bar,
    Colon,
    LeftArrow,
    RightArrow,
    OpenBrace,
    CloseBrace,
    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,
    Equals,
    Plus,
    Minus,
    Asterisk,
    ForwardSlash,
    Gt,
    Lt,
    Tilde,
    Semicolon,
    At,
    Hash,
    Comment(String),
    Unknown(String),
}

impl Token {
    pub fn parse(inp: &str) -> IResult<&str, Token> {
        alt((
            alt((
                map(comment(), |i| Token::Comment(i.to_owned())),
                value(Token::Whitespace, whitespace1()),
                map(float, |f| Token::Number(f as Scalar)),
                alt((
                    value(Token::Graph, tag("graph")),
                    value(Token::Var, tag("var")),
                    value(Token::Let, tag("let")),
                    value(Token::Do, tag("do")),
                    //
                )),
                map(ident_str(), |i| Token::Ident(i.to_owned())),
            )),
            alt((
                value(Token::At, tag("@")),
                value(Token::Hash, tag("#")),
                //
            )),
            alt((
                value(Token::Bar, tag("|")),
                value(Token::CloseBrace, tag("}")),
                value(Token::CloseParen, tag(")")),
                value(Token::CloseBracket, tag("]")),
                value(Token::LeftArrow, tag("<-")),
                value(Token::RightArrow, tag("->")),
                value(Token::Gt, tag(">")),
                value(Token::Lt, tag("<")),
                value(Token::OpenBrace, tag("{")),
                value(Token::OpenParen, tag("(")),
                value(Token::OpenBracket, tag("[")),
                value(Token::Equals, tag("=")),
                value(Token::Plus, tag("+")),
                value(Token::Minus, tag("-")),
                value(Token::Asterisk, tag("*")),
                value(Token::ForwardSlash, tag("/")),
                value(Token::Tilde, tag("~")),
                value(Token::Semicolon, tag(";")),
                value(Token::Colon, tag(":")),
            )),
        ))(inp)
    }

    pub fn many1(inp: &str) -> IResult<&str, Vec<Token>> {
        map(many1(Self::parse), |toks| {
            toks.into_iter()
                .filter(|tok| tok != &Token::Whitespace)
                .collect()
        })(inp)
    }

    pub fn unwrap_ident(self) -> ParsedIdent {
        if let Token::Ident(id) = self {
            ParsedIdent(id)
        } else {
            panic!("expected token to be an ident")
        }
    }

    pub fn unwrap_scalar(self) -> Scalar {
        if let Token::Number(n) = self {
            n as Scalar
        } else {
            panic!("expected token to be a number")
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tokens<'a> {
    pub tokens: &'a [Token],
}

impl<'a> From<&'a [Token]> for Tokens<'a> {
    fn from(value: &'a [Token]) -> Self {
        Self { tokens: value }
    }
}

impl<'a> InputLength for Tokens<'a> {
    fn input_len(&self) -> usize {
        self.tokens.len()
    }
}

impl<'a> InputTake for Tokens<'a> {
    fn take(&self, count: usize) -> Self {
        Self {
            tokens: &self.tokens[0..count],
        }
    }

    fn take_split(&self, count: usize) -> (Self, Self) {
        let (pre, suf) = self.tokens.split_at(count);
        (Self { tokens: suf }, Self { tokens: pre })
    }
}

impl InputLength for Token {
    fn input_len(&self) -> usize {
        1
    }
}

impl<'a> nom::Slice<Range<usize>> for Tokens<'a> {
    fn slice(&self, range: Range<usize>) -> Self {
        Self {
            tokens: self.tokens.slice(range),
        }
    }
}

impl<'a> nom::Slice<RangeTo<usize>> for Tokens<'a> {
    fn slice(&self, range: RangeTo<usize>) -> Self {
        Self {
            tokens: self.tokens.slice(range),
        }
    }
}

impl<'a> nom::Slice<RangeFrom<usize>> for Tokens<'a> {
    fn slice(&self, range: RangeFrom<usize>) -> Self {
        Self {
            tokens: self.tokens.slice(range),
        }
    }
}

impl<'a> nom::Slice<RangeFull> for Tokens<'a> {
    fn slice(&self, range: RangeFull) -> Self {
        Self {
            tokens: self.tokens.slice(range),
        }
    }
}

impl nom::UnspecializedInput for Token {}
impl<'a> nom::UnspecializedInput for Tokens<'a> {}

impl<'a> InputIter for Tokens<'a> {
    type Item = &'a Token;
    type Iter = std::iter::Enumerate<std::slice::Iter<'a, Token>>;
    type IterElem = std::slice::Iter<'a, Token>;

    fn iter_indices(&self) -> Self::Iter {
        self.tokens.iter().enumerate()
    }

    fn iter_elements(&self) -> Self::IterElem {
        self.tokens.iter()
    }

    fn position<P>(&self, predicate: P) -> Option<usize>
    where
        P: Fn(Self::Item) -> bool,
    {
        self.tokens.iter().position(predicate)
    }

    fn slice_index(&self, count: usize) -> Result<usize, nom::Needed> {
        if self.tokens.len() >= count {
            Ok(count)
        } else {
            Err(nom::Needed::Unknown)
        }
    }
}

// impl<'a> nom::Compare<Self> for Tokens<'a> {
//     fn compare(&self, t: Self) -> nom::CompareResult {
//         if self.tokens.len() != t.tokens.len() {
//             nom::CompareResult::Incomplete
//         } else if self.tokens.iter().zip(t.tokens.iter()).all(|(a, b)| a == b) {
//             nom::CompareResult::Ok
//         } else {
//             nom::CompareResult::Error
//         }
//     }

//     fn compare_no_case(&self, t: Self) -> nom::CompareResult {
//         self.compare(t)
//     }
// }

pub fn ident_str<'a>() -> impl FnMut(&'a str) -> IResult<&str, &str> {
    recognize(tuple((
        opt(alt((tag("@"), tag("#")))),
        alpha1,
        opt(many1(alt((alphanumeric1, tag("_"))))),
    )))
}

pub fn comment<'a>() -> impl FnMut(&'a str) -> IResult<&str, &str> {
    map(delimited(tag("/*"), take_until("*/"), tag("*/")), |_| "")
}

pub fn whitespace0<'a>() -> impl FnMut(&'a str) -> IResult<&str, &str> {
    recognize(many0(alt((space1, tag("\n"), tag("\r")))))
}

pub fn whitespace1<'a>() -> impl FnMut(&'a str) -> IResult<&str, &str> {
    recognize(many1(alt((space1, tag("\n"), tag("\r")))))
}

// macro inspired by https://github.com/Rydgel/monkey-rust/blob/master/lib/parser/mod.rs
macro_rules! tag_token {
    ($func:ident, $tag:expr) => {
        #[allow(dead_code)]
        fn $func(tokens: Tokens) -> IResult<Tokens, Tokens> {
            verify(take(1usize), |t: &Tokens| t.tokens[0] == $tag)(tokens)
        }
    };
}

/*
   Bar,
   LeftArrow,
   RightArrow,
   OpenBrace,
   CloseBrace,
   OpenParen,
   CloseParen,
   Equals,
   Plus,
   Minus,
   Asterisk,
   ForwardSlash,
   Gt,
   Lt,
   Tilde,
   Semicolon,
*/

tag_token!(bar, Token::Bar);
tag_token!(leftarrow, Token::LeftArrow);
tag_token!(rightarrow, Token::RightArrow);
tag_token!(openbrace, Token::OpenBrace);
tag_token!(closebrace, Token::CloseBrace);
tag_token!(openparen, Token::OpenParen);
tag_token!(closeparen, Token::CloseParen);
tag_token!(equals, Token::Equals);
tag_token!(plus, Token::Plus);
tag_token!(minus, Token::Minus);
tag_token!(asterisk, Token::Asterisk);
tag_token!(forwardslash, Token::ForwardSlash);
tag_token!(gt, Token::Gt);
tag_token!(lt, Token::Lt);
tag_token!(at, Token::At);
tag_token!(hash, Token::Hash);
tag_token!(tilde, Token::Tilde);
tag_token!(semicolon, Token::Semicolon);
tag_token!(colon, Token::Colon);
tag_token!(openbracket, Token::OpenBracket);
tag_token!(closebracket, Token::CloseBracket);

tag_token!(graph_kw, Token::Graph);
tag_token!(var_kw, Token::Var);
tag_token!(let_kw, Token::Let);
tag_token!(do_kw, Token::Do);

/*
 ===================================================================================
*/

pub fn scalar(inp: Tokens) -> IResult<Tokens, Scalar> {
    map(
        verify(take(1usize), |t: &Tokens| {
            matches!(t.tokens[0], Token::Number(_))
        }),
        |t: Tokens| t.tokens[0].clone().unwrap_scalar(),
    )(inp)
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParsedIdent(pub String);

pub fn ident(tokens: Tokens) -> IResult<Tokens, ParsedIdent> {
    map(
        verify(take(1usize), |t: &Tokens| {
            matches!(t.tokens[0], Token::Ident(_))
        }),
        |t: Tokens| t.tokens[0].clone().unwrap_ident(),
    )(tokens)
}

#[derive(Debug)]
pub enum ParsedCallee {
    ScriptDefined(ParsedIdent),
    Builtin(BuiltinNode, SignalRate),
}

#[derive(Debug)]
pub struct ParsedCall {
    ident: ParsedCallee,
    args: Vec<ParsedExpr>,
}

pub fn call(inp: Tokens) -> IResult<Tokens, ParsedCall> {
    map(
        tuple((ident, delimited(openparen, many0(expr), closeparen))),
        |(ident, args)| {
            if let Some(builtin) = BuiltinNode::try_from_ident(&ident) {
                if ident.0.starts_with('@') {
                    ParsedCall {
                        ident: ParsedCallee::Builtin(builtin, SignalRate::Audio),
                        args,
                    }
                } else {
                    ParsedCall {
                        ident: ParsedCallee::Builtin(builtin, SignalRate::Control),
                        args,
                    }
                }
            } else {
                ParsedCall {
                    ident: ParsedCallee::ScriptDefined(ident),
                    args,
                }
            }
        },
    )(inp)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParsedInfixOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug)]
pub enum ParsedExpr {
    Constant(Scalar),
    Ident(ParsedIdent),
    Infix(InfixExpr),
    Call(ParsedCall),
}

#[derive(Debug)]
pub struct InfixExpr {
    pub lhs: Box<ParsedExpr>,
    pub infix_op: ParsedInfixOp,
    pub rhs: Box<ParsedExpr>,
}

pub fn expr(inp: Tokens) -> IResult<Tokens, ParsedExpr> {
    alt((
        map(scalar, ParsedExpr::Constant),
        map(call, ParsedExpr::Call),
        map(ident, ParsedExpr::Ident),
        map(
            tuple((
                openparen,
                expr,
                alt((
                    value(ParsedInfixOp::Add, plus),
                    value(ParsedInfixOp::Sub, minus),
                    value(ParsedInfixOp::Mul, asterisk),
                    value(ParsedInfixOp::Div, forwardslash),
                )),
                expr,
                closeparen,
            )),
            |(_, lhs, infix_op, rhs, _)| {
                ParsedExpr::Infix(InfixExpr {
                    lhs: lhs.into(),
                    infix_op,
                    rhs: rhs.into(),
                })
            },
        ),
    ))(inp)
}

pub fn list(inp: Tokens) -> IResult<Tokens, Vec<ParsedIdent>> {
    delimited(openbracket, many1(ident), closebracket)(inp)
}

pub fn connection_lhs(inp: Tokens) -> IResult<Tokens, Vec<ParsedIdent>> {
    alt((list, map(ident, |i| vec![i])))(inp)
}

#[derive(Debug)]
pub struct ParsedConnection {
    lhs: Vec<ParsedIdent>,
    rhs: ParsedExpr,
}

pub fn connection(inp: Tokens) -> IResult<Tokens, ParsedConnection> {
    map(
        tuple((connection_lhs, equals, expr, semicolon)),
        |(lhs, _, rhs, _)| ParsedConnection { lhs, rhs },
    )(inp)
}

#[derive(Debug)]
pub struct ParsedLetStatement {
    lhs: Vec<ParsedIdent>,
    rhs: ParsedExpr,
}

pub fn let_statement(inp: Tokens) -> IResult<Tokens, ParsedLetStatement> {
    map(
        preceded(let_kw, tuple((connection_lhs, equals, expr, semicolon))),
        |(lhs, _, rhs, _)| ParsedLetStatement { lhs, rhs },
    )(inp)
}

// pub struct ParsedDoBlock; // todo

#[derive(Debug)]
pub enum ParsedStatement {
    Connection(ParsedConnection),
    Let(ParsedLetStatement),
    // DoBlock(ParsedDoBlock), // todo
}

pub fn statement(inp: Tokens) -> IResult<Tokens, ParsedStatement> {
    alt((
        map(let_statement, ParsedStatement::Let),
        map(connection, ParsedStatement::Connection),
        // todo: ParsedDoBlock
    ))(inp)
}

#[derive(Debug)]
pub struct ParsedInput {
    id: ParsedIdent,
    default: Option<Scalar>,
    minimum: Option<Scalar>,
    maximum: Option<Scalar>,
}

pub fn input(inp: Tokens) -> IResult<Tokens, ParsedInput> {
    map(
        tuple((
            ident,
            opt(delimited(
                openparen,
                tuple((scalar, colon, scalar)),
                closeparen,
            )),
            opt(preceded(equals, scalar)),
        )),
        |(id, bounds, default)| ParsedInput {
            id,
            minimum: bounds.clone().map(|b| b.0),
            maximum: bounds.map(|b| b.2),
            default,
        },
    )(inp)
}

pub fn inputs(inp: Tokens) -> IResult<Tokens, Vec<ParsedInput>> {
    delimited(bar, many0(input), bar)(inp)
}

pub fn outputs(inp: Tokens) -> IResult<Tokens, Vec<ParsedIdent>> {
    delimited(bar, many0(ident), bar)(inp)
}

#[derive(Debug)]
pub struct ParsedSignature {
    inputs: Vec<ParsedInput>,
    outputs: Vec<ParsedIdent>,
}

pub fn signature(inp: Tokens) -> IResult<Tokens, ParsedSignature> {
    map(
        tuple((inputs, rightarrow, outputs)),
        |(inputs, _, outputs)| ParsedSignature { inputs, outputs },
    )(inp)
}

pub fn statements(inp: Tokens) -> IResult<Tokens, Vec<ParsedStatement>> {
    preceded(tilde, delimited(openbrace, many0(statement), closebrace))(inp)
}

#[derive(Debug)]
pub struct ParsedGraph {
    id: ParsedIdent,
    signature: ParsedSignature,
    statements: Vec<ParsedStatement>,
}

pub fn graph(inp: Tokens) -> IResult<Tokens, ParsedGraph> {
    map(
        tuple((
            graph_kw,
            ident,
            delimited(openbrace, tuple((signature, statements)), closebrace),
        )),
        |(_, id, (signature, statements))| ParsedGraph {
            id,
            signature,
            statements,
        },
    )(inp)
}

/*
 ===================================================================================
*/

use petgraph::prelude::*;

use self::builtins::BuiltinNode;

struct SolvedExpr {
    rate: SignalRate,
    ag_idx: Option<(NodeIndex, Vec<usize>)>,
    cg_idx: Option<(NodeIndex, Vec<usize>)>,
}

fn solve_expr(
    graph_id: &ParsedIdent,
    expr: &ParsedExpr,
    super_ag: &mut Graph,
    super_cg: &mut Graph,
    known_graphs: &[ParsedGraph],
    buffer_len: usize,
) -> Result<SolvedExpr, String> {
    match expr {
        ParsedExpr::Constant(value) => {
            let rate = SignalRate::Control; // todo: audio constants
            let node = crate::dsp::basic::Constant::create_node(
                &format!("{value}"),
                rate,
                buffer_len,
                *value,
            );
            match rate {
                SignalRate::Audio => {
                    let idx = super_ag.add_node(node);
                    Ok(SolvedExpr {
                        rate,
                        ag_idx: Some((idx, vec![0])),
                        cg_idx: None,
                    })
                }
                SignalRate::Control => {
                    let idx = super_cg.add_node(node);
                    Ok(SolvedExpr {
                        rate,
                        ag_idx: None,
                        cg_idx: Some((idx, vec![0])),
                    })
                }
            }
        }
        ParsedExpr::Ident(id) => {
            #[allow(clippy::if_same_then_else)]
            let rate = if id.0.starts_with('#') {
                // explicitly a control-rate ident
                SignalRate::Control
            } else if id.0.starts_with('@') {
                // explicitly audio-rate
                SignalRate::Audio
            } else if id.0.starts_with("dac") || id.0.starts_with("adc") {
                // probably audio-rate
                SignalRate::Audio
            } else {
                // default to control-rate (for performance reasons)
                SignalRate::Control
            };
            match rate {
                SignalRate::Audio => {
                    let ag_idx = super_ag.node_id_by_name(&id.0).ok_or(format!("Parsing error (graph `{}`): while parsing expr: audio graph has no node named `{}`", &graph_id.0, &id.0))?;
                    Ok(SolvedExpr {
                        rate,
                        ag_idx: Some((ag_idx, vec![0])), // todo: probably don't assume first output here
                        cg_idx: None,
                    })
                }
                SignalRate::Control => {
                    let cg_idx = super_cg.node_id_by_name(&id.0).ok_or(format!("Parsing error (graph `{}`): while parsing expr: control graph has no node named `{}`", &graph_id.0, &id.0))?;
                    Ok(SolvedExpr {
                        rate,
                        ag_idx: None,
                        cg_idx: Some((cg_idx, vec![0])), // todo: or here
                    })
                }
            }
        }
        ParsedExpr::Infix(InfixExpr { lhs, infix_op, rhs }) => {
            let lhs = solve_expr(graph_id, lhs, super_ag, super_cg, known_graphs, buffer_len)?;
            let rhs = solve_expr(graph_id, rhs, super_ag, super_cg, known_graphs, buffer_len)?;
            // todo: give these nodes actual names
            let op = match infix_op {
                ParsedInfixOp::Add => {
                    crate::dsp::basic::Add::create_node("+", lhs.rate, buffer_len, 0.0, 0.0)
                }
                ParsedInfixOp::Sub => {
                    crate::dsp::basic::Subtract::create_node("-", lhs.rate, buffer_len, 0.0, 0.0)
                }
                ParsedInfixOp::Mul => {
                    crate::dsp::basic::Multiply::create_node("*", lhs.rate, buffer_len, 0.0, 0.0)
                }
                ParsedInfixOp::Div => {
                    crate::dsp::basic::Divide::create_node("/", lhs.rate, buffer_len, 0.0, 0.0)
                }
            };
            match lhs.rate {
                SignalRate::Audio => {
                    let a_idx = lhs.ag_idx.unwrap().0;
                    let b_idx = rhs.ag_idx.unwrap().0;
                    let op_idx = super_ag.add_node(op);

                    super_ag.add_edge(
                        a_idx,
                        op_idx,
                        Connection {
                            source_output: 0,
                            sink_input: 0,
                        },
                    );
                    super_ag.add_edge(
                        b_idx,
                        op_idx,
                        Connection {
                            source_output: 0,
                            sink_input: 1,
                        },
                    );

                    Ok(SolvedExpr {
                        rate: lhs.rate,
                        ag_idx: Some((op_idx, vec![0])),
                        cg_idx: None,
                    })
                }
                SignalRate::Control => {
                    let a_idx = lhs.cg_idx.unwrap().0;
                    let b_idx = rhs.cg_idx.unwrap().0;
                    let op_idx = super_cg.add_node(op);

                    super_cg.add_edge(
                        a_idx,
                        op_idx,
                        Connection {
                            source_output: 0,
                            sink_input: 0,
                        },
                    );
                    super_cg.add_edge(
                        b_idx,
                        op_idx,
                        Connection {
                            source_output: 0,
                            sink_input: 1,
                        },
                    );

                    Ok(SolvedExpr {
                        rate: lhs.rate,
                        ag_idx: None,
                        cg_idx: Some((op_idx, vec![0])),
                    })
                }
            }
        }
        ParsedExpr::Call(ParsedCall { ident, args }) => {
            let mut known_rate = None;
            let (called_an, called_cn) = match ident {
                ParsedCallee::ScriptDefined(ident) => {
                    let graph = known_graphs.iter().find(|g| &g.id == ident).ok_or(format!(
                        "Parsing error (graph `{}`): while parsing expr: undefined reference to graph `{}`",
                        &graph_id.0, &ident.0
                    ))?;
                    let (ag, cg) = solve_graph(graph, buffer_len, known_graphs)?;
                    (Arc::new(ag.into_node()), Arc::new(cg.into_node()))
                }
                ParsedCallee::Builtin(builtin, rate) => {
                    // todo: don't abuse Debug/format here
                    known_rate = Some(*rate);
                    let an = builtin.create_node(
                        &format!("{:?}", builtin),
                        SignalRate::Audio,
                        buffer_len,
                    );
                    let cn = builtin.create_node(
                        &format!("{:?}", builtin),
                        SignalRate::Control,
                        buffer_len,
                    );
                    (an, cn)
                }
            };

            let mut solved_args = vec![];
            for arg in args {
                solved_args.push(solve_expr(
                    graph_id,
                    arg,
                    // todo: called or super???
                    super_ag,
                    super_cg,
                    known_graphs,
                    buffer_len,
                )?);
            }

            let called_ag_idx = super_ag.add_node(called_an);
            let called_cg_idx = super_cg.add_node(called_cn);

            'arg_iter: for (arg_id, arg) in solved_args.into_iter().enumerate() {
                if let Some(known_rate) = known_rate {
                    match (arg.rate, known_rate) {
                        (SignalRate::Control, SignalRate::Audio) => {
                            // insert auto-c2a
                            let (c2a_an, c2a_cn) = ControlToAudio::create_nodes(
                                &format!("auto_c2a_{:?}_{}", ident, arg_id),
                                buffer_len,
                            );
                            let c2a_an = super_ag.add_node(c2a_an);
                            let c2a_cn = super_cg.add_node(c2a_cn);
                            let (arg_cg_idx, cg_from_outs) = arg.cg_idx.unwrap();
                            if cg_from_outs.len() != 1 {
                                return Err(format!("Parsing error (graph `{}`): expected expression to have a single output (has {})", &graph_id.0, cg_from_outs.len()));
                            }
                            super_cg.add_edge(
                                arg_cg_idx,
                                c2a_cn,
                                Connection {
                                    source_output: cg_from_outs[0],
                                    sink_input: 0,
                                },
                            );
                            super_ag.add_edge(
                                c2a_an,
                                called_ag_idx,
                                Connection {
                                    source_output: 0,
                                    sink_input: arg_id,
                                },
                            );

                            continue 'arg_iter;
                        }
                        (SignalRate::Audio, SignalRate::Control) => {
                            todo!("insert auto-a2c here");
                        }
                        _ => {}
                    }
                } else {
                    known_rate = Some(arg.rate);
                }

                assert!(
                    !(arg.ag_idx.is_some() && arg.cg_idx.is_some()),
                    "todo: exprs using both graphs"
                );
                if let Some((ag_from_idx, ag_from_outs)) = arg.ag_idx {
                    if ag_from_outs.len() != 1 {
                        return Err(format!("Parsing error (graph `{}`): expected expression to have a single output (has {})", &graph_id.0, ag_from_outs.len()));
                    }
                    // todo: don't assume it's the first input, probably
                    super_ag.add_edge(
                        ag_from_idx,
                        called_ag_idx,
                        Connection {
                            source_output: 0,
                            sink_input: arg_id,
                        },
                    );
                }
                if let Some((cg_from_idx, cg_from_outs)) = arg.cg_idx {
                    if cg_from_outs.len() != 1 {
                        return Err(format!("Parsing error (graph `{}`): expected expression to have a single output (has {})", &graph_id.0, cg_from_outs.len()));
                    }
                    // todo: don't assume it's the first input, probably
                    super_cg.add_edge(
                        cg_from_idx,
                        called_cg_idx,
                        Connection {
                            source_output: 0,
                            sink_input: arg_id,
                        },
                    );
                }
            }

            let known_rate = known_rate.unwrap();

            let (ag_idx, cg_idx) = match known_rate {
                SignalRate::Audio => {
                    // todo: remove called control node from supergraph
                    (Some((called_ag_idx, vec![0])), None)
                }
                SignalRate::Control => {
                    // todo: remove called audio node from supergraph
                    (None, Some((called_cg_idx, vec![0])))
                }
            };

            Ok(SolvedExpr {
                ag_idx,
                cg_idx,
                rate: known_rate,
            })
        }
    }
}

fn solve_graph(
    graph: &ParsedGraph,
    buffer_len: usize,
    known_graphs: &[ParsedGraph],
) -> Result<(Graph, Graph), String> {
    let ParsedGraph {
        signature: ParsedSignature { inputs, outputs },
        statements,
        id,
    } = graph;

    let inputs = inputs
        .iter()
        .map(|inp| Input {
            name: inp.id.0.clone(),
            minimum: inp.minimum.map(Signal::new),
            maximum: inp.maximum.map(Signal::new),
            default: inp.default.map(Signal::new),
            is_ui: graph.id.0 == "main", // FIXME: hacky
            implicit: false,
        })
        .collect::<Vec<_>>();
    let outputs = outputs
        .iter()
        .map(|out| Output {
            name: out.0.to_owned(),
        })
        .collect::<Vec<_>>();

    let mut ag = Graph::new(
        Some(NodeName::new(&id.0)),
        SignalRate::Audio,
        buffer_len,
        // todo: somehow auto-distinguish audio/control inputs/outputs
        inputs.clone(),
        outputs.clone(),
    );
    let mut cg = Graph::new(
        Some(NodeName::new(&id.0)),
        SignalRate::Control,
        buffer_len,
        inputs,
        outputs,
    );

    for stmt in statements {
        match stmt {
            ParsedStatement::Let(stmt) => {
                let rhs = solve_expr(id, &stmt.rhs, &mut ag, &mut cg, known_graphs, buffer_len)?;
                assert!(
                    !(rhs.ag_idx.is_some() && rhs.cg_idx.is_some()),
                    "todo: exprs using both graphs"
                );

                if let Some((ag_from_idx, ag_from_outs)) = rhs.ag_idx {
                    if ag_from_outs.len() != stmt.lhs.len() {
                        return Err(format!("Parsing error (graph `{}`): in `let` statement, expected as many left-hand terms as the right-hand expression has outputs ({} vs {})", &id.0, stmt.lhs.len(), ag_from_outs.len()));
                    }
                    for (to, ag_from_out) in stmt.lhs.iter().zip(ag_from_outs.iter()) {
                        // this is where `let` statements differ from normal connections
                        let node =
                            LetBinding::create_node(&to.0, SignalRate::Audio, buffer_len, 0.0);
                        let to_idx = ag.add_node(node);
                        // todo: don't assume it's the first input, probably
                        ag.add_edge(
                            ag_from_idx,
                            to_idx,
                            Connection {
                                source_output: *ag_from_out,
                                sink_input: 0,
                            },
                        );
                    }
                }
                if let Some((cg_from_idx, cg_from_outs)) = rhs.cg_idx {
                    if cg_from_outs.len() != stmt.lhs.len() {
                        return Err(format!("Parsing error (graph `{}`): expected as many left-hand terms as the right-hand expression has outputs ({} vs {})", &id.0, stmt.lhs.len(), cg_from_outs.len()));
                    }
                    for (to, cg_from_out) in stmt.lhs.iter().zip(cg_from_outs.iter()) {
                        // this is where `let` statements differ from normal connections
                        let node =
                            LetBinding::create_node(&to.0, SignalRate::Control, buffer_len, 0.0);
                        let to_idx = cg.add_node(node);
                        // todo: don't assume it's the first input, probably
                        cg.add_edge(
                            cg_from_idx,
                            to_idx,
                            Connection {
                                source_output: *cg_from_out,
                                sink_input: 0,
                            },
                        );
                    }
                }
            }
            ParsedStatement::Connection(conn) => {
                let rhs = solve_expr(id, &conn.rhs, &mut ag, &mut cg, known_graphs, buffer_len)?;
                assert!(
                    !(rhs.ag_idx.is_some() && rhs.cg_idx.is_some()),
                    "todo: exprs using both graphs"
                );
                if let Some((ag_from_idx, ag_from_outs)) = rhs.ag_idx {
                    if ag_from_outs.len() != conn.lhs.len() {
                        return Err(format!("Parsing error (graph `{}`): expected as many left-hand terms as the right-hand expression has outputs ({} vs {})", &id.0, conn.lhs.len(), ag_from_outs.len()));
                    }
                    for (to, ag_from_out) in conn.lhs.iter().zip(ag_from_outs.iter()) {
                        let to_idx = ag.node_id_by_name(&to.0).ok_or(format!(
                            "Parsing error (graph `{}`): audio graph has no node named `{}`",
                            &id.0, &to.0
                        ))?;
                        // todo: don't assume it's the first input, probably
                        ag.add_edge(
                            ag_from_idx,
                            to_idx,
                            Connection {
                                source_output: *ag_from_out,
                                sink_input: 0,
                            },
                        );
                    }
                }
                if let Some((cg_from_idx, cg_from_outs)) = rhs.cg_idx {
                    if cg_from_outs.len() != conn.lhs.len() {
                        return Err(format!("Parsing error (graph `{}`): expected as many left-hand terms as the right-hand expression has outputs ({} vs {})", &id.0, conn.lhs.len(), cg_from_outs.len()));
                    }
                    for (to, cg_from_out) in conn.lhs.iter().zip(cg_from_outs.iter()) {
                        let to_idx = cg.node_id_by_name(&to.0).ok_or(format!(
                            "Parsing error (graph `{}`): control graph has no node named `{}`",
                            &id.0, &to.0
                        ))?;
                        // todo: don't assume it's the first input, probably
                        cg.add_edge(
                            cg_from_idx,
                            to_idx,
                            Connection {
                                source_output: *cg_from_out,
                                sink_input: 0,
                            },
                        );
                    }
                }
            }
        }
    }

    ag.write_dot(&format!("{}_audio.dot", &id.0));
    cg.write_dot(&format!("{}_control.dot", &id.0));

    Ok((ag, cg))
}

/// Main parsing fn
pub fn parse_script(inp: &str, buffer_len: usize) -> Result<(Graph, Graph), String> {
    let (garbage, tokens) = Token::many1(inp).unwrap();
    if !garbage.is_empty() {
        return Err(format!("Parsing error: unexpected garbage:\n{garbage}"));
    }
    let tokens = Tokens { tokens: &tokens };
    let (garbage, graphs) = many1(graph)(tokens).unwrap();
    if !garbage.tokens.is_empty() {
        return Err(format!(
            "Parsing error: unexpected garbage:\n{:?}",
            garbage.tokens
        ));
    }

    // dbg!(&graphs);

    let main_graph = graphs
        .iter()
        .find(|g| g.id.0 == "main")
        .ok_or("Parsing error: `main` graph not found".to_owned())?;
    solve_graph(main_graph, buffer_len, &graphs)
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     #[test]
//     fn test_parse() {
//         let (ag, cg) = parse_script(include_str!("../../test-scripts/syntax2.papr"), 1024).unwrap();
//     }
// }
