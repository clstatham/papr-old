use std::{collections::BTreeMap, fs::File, io::Read, path::Path, sync::Arc};

use miette::{miette, Context, Diagnostic, Result, SourceOffset};
use pest::{
    error::ErrorVariant,
    iterators::Pairs,
    pratt_parser::{Assoc, Op, PrattParser},
    Parser, Position,
};
use pest_derive::Parser;
use thiserror::Error;

use crate::{
    dsp::{
        basic::{AudioToControl, ControlToAudio},
        graph_util::LetBinding,
        Signal,
    },
    graph::{Connection, Graph, Input, NodeName, Output},
    Scalar,
};

use self::builtins::BuiltinNode;

pub mod builtins;
#[cfg(test)]
mod tests;

#[derive(Error, Diagnostic, Debug)]
#[error("parsing error: {msg}")]
pub struct ParseError {
    #[source_code]
    src: String,

    #[label("here")]
    span: SourceOffset,

    msg: String,
}

impl ParseError {
    pub fn new_from_pos(src: &str, variant: ErrorVariant<Rule>, pos: Position<'_>) -> Self {
        Self::from_source_and_pest_error(src, pest::error::Error::new_from_pos(variant, pos))
    }

    pub fn from_source_and_pest_error(src: &str, value: pest::error::Error<Rule>) -> Self {
        Self {
            src: src.to_owned(),
            span: match value.location {
                pest::error::InputLocation::Pos(pos) => pos.into(),
                pest::error::InputLocation::Span(sp) => sp.0.into(),
            },
            msg: value.variant.message().into(),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ParsedSignalRate {
    Audio,
    Control,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ParsedIdent(pub String, Option<ParsedSignalRate>);

impl ParsedIdent {
    pub fn into_parsed_callee(self) -> ParsedCallee {
        if let Some(node) = BuiltinNode::try_from_ident(&self) {
            ParsedCallee::Builtin(node, self.1)
        } else {
            ParsedCallee::ScriptDefined(self)
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ParsedImport(String);

#[derive(Debug, Clone)]
pub enum ParsedCallee {
    ScriptDefined(ParsedIdent),
    Builtin(BuiltinNode, Option<ParsedSignalRate>),
}

#[derive(Debug, Clone)]
pub enum ParsedCreationArg {
    Scalar(Scalar),
    String(String),
}

impl ParsedCreationArg {
    pub fn unwrap_string(&self) -> &str {
        if let Self::String(s) = self {
            s
        } else {
            panic!("expected creation argument to be a string, got {:?}", self);
        }
    }

    pub fn unwrap_scalar(&self) -> &Scalar {
        if let Self::Scalar(s) = self {
            s
        } else {
            panic!("expected creation argument to be a scalar, got {:?}", self);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParsedCall {
    callee: ParsedCallee,
    creation_args: Vec<ParsedCreationArg>,
    args: Vec<ParsedExpr>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ParsedInfixOp {
    Add,
    Sub,
    Mul,
    Div,
    Gt,
    Lt,
    Eq,
    Neq,
    And,
    Or,
    Xor,
    Rem,
}

#[derive(Debug, Clone)]
pub enum ParsedExpr {
    Constant(Scalar),
    Ident(ParsedIdent),
    Infix(ParsedInfixExpr),
    Call(ParsedCall),
}

#[derive(Debug, Clone)]
pub struct ParsedInfixExpr {
    pub lhs: Box<ParsedExpr>,
    pub infix_op: ParsedInfixOp,
    pub rhs: Box<ParsedExpr>,
}

#[derive(Debug, Clone)]
pub struct ParsedConnection {
    lhs: Vec<ParsedIdent>,
    rhs: ParsedExpr,
}

#[derive(Debug, Clone)]
pub struct ParsedLetStatement {
    lhs: Vec<ParsedIdent>,
    rhs: ParsedExpr,
}

#[derive(Debug, Clone)]
pub enum ParsedStatement {
    Let(ParsedLetStatement),
    Connection(ParsedConnection),
}

#[derive(Debug, Clone)]
pub struct ParsedInput {
    id: ParsedIdent,
    default: Option<Scalar>,
    minimum: Option<Scalar>,
    maximum: Option<Scalar>,
}

#[derive(Debug, Clone)]
pub struct ParsedSignature {
    inputs: Vec<ParsedInput>,
    outputs: Vec<ParsedIdent>,
}

#[derive(Debug, Clone)]
pub struct ParsedGraph {
    id: ParsedIdent,
    signature: ParsedSignature,
    statements: Vec<ParsedStatement>,
}

#[derive(Parser)]
#[grammar = "src/parser3/papr.pest"]
pub struct PaprParser;

use pest::iterators::Pair;

pub fn parse_signal_rate(src: &str, inp: Pair<Rule>) -> Result<ParsedSignalRate> {
    if inp.as_rule() == Rule::signal_rate {
        match inp.as_str() {
            "@" => Ok(ParsedSignalRate::Audio),
            "#" => Ok(ParsedSignalRate::Control),
            _ => unreachable!(),
        }
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: format!(
                    "Expected signal rate (either `@` or `#`)\nGot {:?}",
                    inp.as_rule()
                ),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_ident(src: &str, inp: Pair<Rule>) -> Result<ParsedIdent> {
    if let Rule::ident = inp.as_rule() {
        let mut inner = inp.into_inner();
        match inner.next() {
            Some(rate) if rate.as_rule() == Rule::signal_rate => match inner.next() {
                Some(id) if id.as_rule() == Rule::ident_raw => Ok(ParsedIdent(
                    id.as_str().to_owned(),
                    Some(parse_signal_rate(src, rate)?),
                )),
                _ => Err(ParseError::new_from_pos(
                    src,
                    ErrorVariant::CustomError {
                        message: "Expected identifier after signal rate".to_owned(),
                    },
                    rate.as_span().start_pos(),
                )
                .into()),
            },
            Some(name) if name.as_rule() == Rule::ident_raw => {
                Ok(ParsedIdent(name.as_str().to_owned(), None))
            }
            _ => unreachable!("BUG: Internal parsing error: failed to parse ident"),
        }
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: "Expected identifier".to_owned(),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_graph_name(src: &str, inp: Pair<Rule>) -> Result<ParsedCallee> {
    if let Rule::graph_name = inp.as_rule() {
        let mut inner = inp.into_inner();
        match inner.next() {
            Some(rate) if rate.as_rule() == Rule::signal_rate => match inner.next() {
                Some(id) if id.as_rule() == Rule::graph_name_raw => Ok(ParsedIdent(
                    id.as_str().to_owned(),
                    Some(parse_signal_rate(src, rate)?),
                )
                .into_parsed_callee()),
                _ => Err(ParseError::new_from_pos(
                    src,
                    ErrorVariant::CustomError {
                        message: "Expected graph name after signal rate".to_owned(),
                    },
                    rate.as_span().start_pos(),
                )
                .into()),
            },
            Some(name) if name.as_rule() == Rule::graph_name_raw => {
                Ok(ParsedIdent(name.as_str().to_owned(), None).into_parsed_callee())
            }
            _ => unreachable!("BUG: Internal parsing error: failed to parse graph name"),
        }
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: "Expected graph name".to_owned(),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_scalar(src: &str, inp: Pair<Rule>) -> Result<Scalar> {
    if let Rule::scalar = inp.as_rule() {
        let val = inp
            .as_str()
            .parse::<Scalar>()
            .expect("BUG: Internal parsing error: failed to parse scalar");
        Ok(val)
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: "Expected number".to_owned(),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_string(src: &str, inp: Pair<Rule>) -> Result<String> {
    if inp.as_rule() == Rule::string {
        // trim the quotes
        // todo: remove this panic
        Ok(inp.into_inner().next().unwrap().as_str().to_owned())
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: "Expected string".to_owned(),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_import(src: &str, inp: Pair<Rule>) -> Result<String> {
    if inp.as_rule() == Rule::import_stmt {
        Ok(parse_string(src, inp.into_inner().next().unwrap())?)
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: "Expected import statement".to_owned(),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_call(src: &str, inp: Pair<Rule>) -> Result<ParsedCall> {
    if inp.as_rule() == Rule::call {
        let mut inner = inp.into_inner();
        let callee = parse_graph_name(src, inner.next().unwrap())?;
        let mut creation_args = vec![];
        if let Some(Rule::creation_args) = inner.peek().map(|i| i.as_rule()) {
            let ca = inner.next().unwrap();
            for ca in ca.into_inner() {
                if ca.as_rule() == Rule::scalar {
                    creation_args.push(ParsedCreationArg::Scalar(parse_scalar(src, ca)?));
                } else if ca.as_rule() == Rule::string {
                    creation_args.push(ParsedCreationArg::String(parse_string(src, ca)?));
                } else {
                    unreachable!("BUG: Internal parsing error: failed to parse creation arg");
                }
            }
        }
        let mut args = vec![];
        for arg in inner {
            args.push(parse_expr(src, arg.into_inner())?);
        }
        Ok(ParsedCall {
            callee,
            creation_args,
            args,
        })
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: "Expected graph instantiation".to_owned(),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_expr(src: &str, inp: Pairs<Rule>) -> Result<ParsedExpr> {
    let pratt = PrattParser::new()
        .op(Op::infix(Rule::add, Assoc::Left) | Op::infix(Rule::sub, Assoc::Left))
        .op(Op::infix(Rule::mul, Assoc::Left) | Op::infix(Rule::div, Assoc::Left))
        .op(Op::infix(Rule::rem, Assoc::Left))
        .op(Op::infix(Rule::gt, Assoc::Left) | Op::infix(Rule::lt, Assoc::Left))
        .op(Op::infix(Rule::eq, Assoc::Left) | Op::infix(Rule::neq, Assoc::Left))
        .op(Op::infix(Rule::and, Assoc::Left))
        .op(Op::infix(Rule::or, Assoc::Left))
        .op(Op::infix(Rule::xor, Assoc::Left));

    pratt
        .map_primary(|primary| match primary.as_rule() {
            Rule::ident => Ok(ParsedExpr::Ident(parse_ident(src, primary)?)),
            Rule::call => Ok(ParsedExpr::Call(parse_call(src, primary)?)),
            Rule::scalar => Ok(ParsedExpr::Constant(parse_scalar(src, primary)?)),
            Rule::expr => parse_expr(src, primary.into_inner()),
            rule => unreachable!("BUG: Internal parsing error: failed to parse expr: {rule:?}"),
        })
        .map_infix(|lhs, op, rhs| {
            let infix_op = parse_infix_op(src, op)?;
            Ok(ParsedExpr::Infix(ParsedInfixExpr {
                lhs: Box::new(lhs?),
                infix_op,
                rhs: Box::new(rhs?),
            }))
        })
        .parse(inp)
}

// pub fn parse_infix_expr(src: &str, inp: Pair<Rule>) -> Result<ParsedInfixExpr> {
//     if inp.as_rule() == Rule::infix_expr {
//         let mut inner = inp.into_inner();
//         let lhs = parse_expr(src, inner.next().unwrap())?;
//         let infix_op = parse_infix_op(src, inner.next().unwrap())?;
//         let rhs = parse_expr(src, inner.next().unwrap())?;
//         Ok(ParsedInfixExpr {
//             lhs: Box::new(lhs),
//             infix_op,
//             rhs: Box::new(rhs),
//         })
//     } else {
//         Err(ParseError::new_from_pos(
//             src,
//             ErrorVariant::CustomError {
//                 message: "Expected infix expression".to_owned(),
//             },
//             inp.as_span().start_pos(),
//         )
//         .into())
//     }
// }

pub fn parse_infix_op(src: &str, inp: Pair<Rule>) -> Result<ParsedInfixOp> {
    match inp.as_str() {
        "+" => Ok(ParsedInfixOp::Add),
        "-" => Ok(ParsedInfixOp::Sub),
        "*" => Ok(ParsedInfixOp::Mul),
        "/" => Ok(ParsedInfixOp::Div),
        "&" => Ok(ParsedInfixOp::And),
        "|" => Ok(ParsedInfixOp::Or),
        "^" => Ok(ParsedInfixOp::Xor),
        "==" => Ok(ParsedInfixOp::Eq),
        "!=" => Ok(ParsedInfixOp::Neq),
        ">" => Ok(ParsedInfixOp::Gt),
        "<" => Ok(ParsedInfixOp::Lt),
        "%" => Ok(ParsedInfixOp::Rem),
        _ => Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: "Expected infix operator".to_owned(),
            },
            inp.as_span().start_pos(),
        )
        .into()),
    }
}

pub fn parse_list(src: &str, inp: Pair<Rule>) -> Result<Vec<ParsedIdent>> {
    if inp.as_rule() == Rule::list {
        let mut list = vec![];
        for inp in inp.into_inner() {
            list.push(parse_ident(src, inp)?);
        }
        Ok(list)
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: "Expected list of identifiers".to_owned(),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_connection(src: &str, inp: Pair<Rule>) -> Result<ParsedConnection> {
    if inp.as_rule() == Rule::connection {
        let mut inner = inp.into_inner();
        let lhs = inner.next().unwrap();
        let lhs = if lhs.as_rule() == Rule::connection_lhs {
            let lhs = lhs.into_inner().next().unwrap();
            match lhs.as_rule() {
                Rule::list => parse_list(src, lhs)?,
                Rule::ident => vec![parse_ident(src, lhs)?],
                _ => unreachable!(),
            }
        } else {
            unreachable!()
        };
        let expr = parse_expr(src, inner)?;
        Ok(ParsedConnection { lhs, rhs: expr })
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: format!("Expected connection\nGot {:?}", inp.as_rule()),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_statement(src: &str, inp: Pair<Rule>) -> Result<ParsedStatement> {
    if inp.as_rule() == Rule::statement {
        let next = inp.into_inner().next().unwrap();
        match next.as_rule() {
            Rule::connection => Ok(ParsedStatement::Connection(parse_connection(src, next)?)),
            Rule::let_statement => {
                let connection = next.into_inner().next().unwrap();
                let connection = parse_connection(src, connection)?;
                Ok(ParsedStatement::Let(ParsedLetStatement {
                    lhs: connection.lhs,
                    rhs: connection.rhs,
                }))
            }
            _ => unreachable!(),
        }
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: "Expected statement".to_owned(),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_statement_block(src: &str, inp: Pair<Rule>) -> Result<Vec<ParsedStatement>> {
    if inp.as_rule() == Rule::statement_block {
        let mut stmts = vec![];
        for inp in inp.into_inner() {
            stmts.push(parse_statement(src, inp)?);
        }
        Ok(stmts)
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: "Expected statement block (`~ { ... }`)".to_owned(),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_inputs(src: &str, inp: Pair<Rule>) -> Result<Vec<ParsedInput>> {
    if inp.as_rule() == Rule::inputs {
        let mut ins = vec![];
        for inp in inp.into_inner() {
            ins.push(ParsedInput {
                id: parse_ident(src, inp)?,
                default: None,
                minimum: None,
                maximum: None,
            });
        }
        Ok(ins)
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: "Expected inputs".to_owned(),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_outputs(src: &str, inp: Pair<Rule>) -> Result<Vec<ParsedIdent>> {
    if inp.as_rule() == Rule::outputs {
        let mut ins = vec![];
        for inp in inp.into_inner() {
            ins.push(parse_ident(src, inp)?);
        }
        Ok(ins)
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: "Expected outputs".to_owned(),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_signature(src: &str, inp: Pair<Rule>) -> Result<ParsedSignature> {
    if inp.as_rule() == Rule::signature {
        let mut inner = inp.into_inner();
        let inputs = parse_inputs(src, inner.next().unwrap())?;
        let outputs = parse_outputs(src, inner.next().unwrap())?;
        Ok(ParsedSignature { inputs, outputs })
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: "Expected signature (`|...| -> |...|`)".to_owned(),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_graph_def(src: &str, inp: Pair<Rule>) -> Result<ParsedGraph> {
    if inp.as_rule() == Rule::graph_def {
        let mut inner = inp.into_inner();
        let graph_name = parse_graph_name(src, inner.next().unwrap())?;
        let signature = parse_signature(src, inner.next().unwrap())?;
        let statements = parse_statement_block(src, inner.next().unwrap())?;
        if let ParsedCallee::ScriptDefined(id) = graph_name {
            Ok(ParsedGraph {
                id,
                signature,
                statements,
            })
        } else {
            unreachable!()
        }
    } else {
        Err(ParseError::new_from_pos(
            src,
            ErrorVariant::CustomError {
                message: format!("Expected graph definition\nGot {:?}", inp.as_rule()),
            },
            inp.as_span().start_pos(),
        )
        .into())
    }
}

pub fn parse_script(path: &Path, known_graphs: &mut BTreeMap<String, ParsedGraph>) -> Result<()> {
    let mut f = File::open(path).unwrap();
    let mut src = String::new();
    f.read_to_string(&mut src).unwrap();

    let mut papr = PaprParser::parse(Rule::main, &src)
        .map_err(|e| ParseError::from_source_and_pest_error(&src, e))?
        .next()
        .unwrap()
        .into_inner();
    if let Some(peek) = papr.peek() {
        if peek.as_rule() == Rule::import_stmt {
            while let Ok(imp) = parse_import(&src, papr.peek().unwrap()) {
                let mut new_path = path.parent().unwrap().to_path_buf();
                new_path.push(imp);
                parse_script(&new_path, known_graphs)?;
                papr.next();
            }
        }
    }

    for graph_def in papr {
        if graph_def.as_rule() == Rule::graph_def {
            let graph = parse_graph_def(&src, graph_def)?;
            known_graphs.insert(graph.id.0.to_string(), graph.clone());
        }
    }

    Ok(())
}

pub fn parse_main_script(path: &Path) -> Result<(Graph, Graph)> {
    let mut known_graphs = BTreeMap::default();

    parse_script(path, &mut known_graphs)?;
    let main = known_graphs
        .get("Main")
        .ok_or(miette!("Failed to find `Main` graph"))?
        .clone();

    solve_graph(&main, &mut known_graphs)
}

use petgraph::prelude::*;

struct SolvedExpr {
    rate: ParsedSignalRate,
    ag_idx: Option<(NodeIndex, Vec<usize>)>,
    cg_idx: Option<(NodeIndex, Vec<usize>)>,
}

fn solve_expr(
    graph_id: &ParsedIdent,
    expr: &ParsedExpr,
    super_ag: &mut Graph,
    super_cg: &mut Graph,
    known_graphs: &mut BTreeMap<String, ParsedGraph>,
    lhs_ident: Option<&ParsedIdent>,
) -> Result<SolvedExpr> {
    match expr {
        ParsedExpr::Constant(value) => {
            let rate = graph_id.1.unwrap_or(ParsedSignalRate::Control);
            let node = crate::dsp::basic::Constant::create_node(&format!("{value}"), *value);
            match rate {
                ParsedSignalRate::Audio => {
                    let idx = super_ag.add_node(node);
                    Ok(SolvedExpr {
                        rate,
                        ag_idx: Some((idx, vec![0])),
                        cg_idx: None,
                    })
                }
                ParsedSignalRate::Control => {
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
            let rate = id.1.or(graph_id.1).ok_or_else(|| {
                miette!(
                    "Parsing error (graph `{}`): couldn't determine signal rate of identifier `{}`",
                    &graph_id.0,
                    &id.0
                )
            })?;
            match rate {
                ParsedSignalRate::Audio => {
                    let ag_idx = super_ag
                        .node_id_by_name(&id.0)
                        .context("while parsing expr")?;
                    Ok(SolvedExpr {
                        rate,
                        ag_idx: Some((ag_idx, vec![0])), // todo: probably don't assume first output here
                        cg_idx: None,
                    })
                }
                ParsedSignalRate::Control => {
                    let cg_idx = super_cg
                        .node_id_by_name(&id.0)
                        .context("while parsing expr")?;
                    Ok(SolvedExpr {
                        rate,
                        ag_idx: None,
                        cg_idx: Some((cg_idx, vec![0])), // todo: or here
                    })
                }
            }
        }
        ParsedExpr::Infix(ParsedInfixExpr { lhs, infix_op, rhs }) => {
            let lhs = solve_expr(graph_id, lhs, super_ag, super_cg, known_graphs, None)?;
            let rhs = solve_expr(graph_id, rhs, super_ag, super_cg, known_graphs, None)?;
            // todo: give these nodes actual names
            let op = match infix_op {
                ParsedInfixOp::Add => crate::dsp::basic::Add::create_node("+"),
                ParsedInfixOp::Sub => crate::dsp::basic::Sub::create_node("-"),
                ParsedInfixOp::Mul => crate::dsp::basic::Mul::create_node("*"),
                ParsedInfixOp::Div => crate::dsp::basic::Div::create_node("/"),
                ParsedInfixOp::Gt => crate::dsp::basic::Gt::create_node(">"),
                ParsedInfixOp::Lt => crate::dsp::basic::Lt::create_node("<"),
                ParsedInfixOp::Eq => crate::dsp::basic::Eq::create_node("=="),
                ParsedInfixOp::Neq => crate::dsp::basic::Div::create_node("!="),
                ParsedInfixOp::And => crate::dsp::basic::And::create_node("&"),
                ParsedInfixOp::Or => crate::dsp::basic::Or::create_node("|"),
                ParsedInfixOp::Xor => crate::dsp::basic::Xor::create_node("^"),
                ParsedInfixOp::Rem => crate::dsp::basic::Rem::create_node("%"),
            };
            match (lhs.rate, rhs.rate) {
                (ParsedSignalRate::Audio, ParsedSignalRate::Audio) => {
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
                (ParsedSignalRate::Control, ParsedSignalRate::Control) => {
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
                (ParsedSignalRate::Audio, ParsedSignalRate::Control) => {
                    let a_idx_ag = lhs.ag_idx.unwrap().0;
                    let b_idx_cg = rhs.cg_idx.unwrap().0;
                    let op_idx = super_ag.add_node(op);
                    let (c2a_an, c2a_cn) = ControlToAudio::create_nodes(&format!(
                        "auto_c2a_{:?}_{:?}",
                        a_idx_ag, b_idx_cg
                    ));
                    let c2a_idx_ag = super_ag.add_node(c2a_an);
                    let c2a_idx_cg = super_cg.add_node(c2a_cn);

                    super_ag.add_edge(
                        a_idx_ag,
                        op_idx,
                        Connection {
                            source_output: 0,
                            sink_input: 0,
                        },
                    );

                    super_cg.add_edge(
                        b_idx_cg,
                        c2a_idx_cg,
                        Connection {
                            source_output: 0,
                            sink_input: 0,
                        },
                    );

                    super_ag.add_edge(
                        c2a_idx_ag,
                        op_idx,
                        Connection {
                            source_output: 0,
                            sink_input: 1,
                        },
                    );

                    Ok(SolvedExpr {
                        rate: ParsedSignalRate::Audio,
                        ag_idx: Some((op_idx, vec![0])),
                        cg_idx: None,
                    })
                }
                (ParsedSignalRate::Control, ParsedSignalRate::Audio) => {
                    let a_idx_cg = lhs.cg_idx.unwrap().0;
                    let b_idx_ag = rhs.ag_idx.unwrap().0;
                    let op_idx = super_cg.add_node(op);
                    let (a2c_an, a2c_cn) = AudioToControl::create_nodes(&format!(
                        "auto_a2c_{:?}_{:?}",
                        a_idx_cg, b_idx_ag
                    ));
                    let a2c_idx_ag = super_ag.add_node(a2c_an);
                    let a2c_idx_cg = super_cg.add_node(a2c_cn);

                    super_cg.add_edge(
                        a_idx_cg,
                        op_idx,
                        Connection {
                            source_output: 0,
                            sink_input: 0,
                        },
                    );

                    super_ag.add_edge(
                        b_idx_ag,
                        a2c_idx_ag,
                        Connection {
                            source_output: 0,
                            sink_input: 0,
                        },
                    );

                    super_cg.add_edge(
                        a2c_idx_cg,
                        op_idx,
                        Connection {
                            source_output: 0,
                            sink_input: 1,
                        },
                    );

                    Ok(SolvedExpr {
                        rate: ParsedSignalRate::Control,
                        ag_idx: None,
                        cg_idx: Some((op_idx, vec![0])),
                    })
                }
            }
        }
        ParsedExpr::Call(ParsedCall {
            callee,
            creation_args,
            args,
        }) => {
            let mut known_rate;
            let n_outs;
            let (called_an, called_cn) = match callee {
                ParsedCallee::ScriptDefined(ident) => {
                    known_rate = ident.1;
                    let mut graph = known_graphs.get(&ident.0).ok_or(miette!(
                        "Parsing error (graph `{}`): while parsing expr: undefined reference to graph `{}`",
                        &graph_id.0, &ident.0
                    ))?.clone();
                    graph.id.1 = known_rate;
                    n_outs = graph.signature.outputs.len();
                    let (ag, cg) = solve_graph(&graph, known_graphs)?;
                    (Arc::new(ag.into_node()), Arc::new(cg.into_node()))
                }
                ParsedCallee::Builtin(builtin, rate) => {
                    // todo: don't abuse Debug/format here
                    known_rate = rate.or(graph_id.1);
                    let lhs_ident = lhs_ident
                        .map(|id| id.0.to_owned())
                        .unwrap_or(format!("{:?}", builtin));
                    let an = builtin.create_node(&lhs_ident, creation_args)?;
                    let cn = builtin.create_node(&lhs_ident, creation_args)?;
                    n_outs = match known_rate {
                        Some(ParsedSignalRate::Audio) => an.outputs.len(),
                        Some(ParsedSignalRate::Control) => cn.outputs.len(),
                        _ => todo!(),
                    };
                    (an, cn)
                }
            };

            let mut solved_args = vec![];
            for arg in args {
                solved_args.push(solve_expr(
                    graph_id,
                    arg,
                    super_ag,
                    super_cg,
                    known_graphs,
                    None,
                )?);
            }

            let called_ag_idx = super_ag.add_node(called_an);
            let called_cg_idx = super_cg.add_node(called_cn);

            let mut audio_arg_count = 0;
            let mut control_arg_count = 0;
            'arg_iter: for (arg_id, arg) in solved_args.into_iter().enumerate() {
                if let Some(known_rate) = known_rate {
                    match (arg.rate, known_rate) {
                        (ParsedSignalRate::Control, ParsedSignalRate::Audio) => {
                            // insert auto-c2a
                            let (c2a_an, c2a_cn) = ControlToAudio::create_nodes(&format!(
                                "auto_c2a_{:?}_{}",
                                callee, arg_id
                            ));
                            let c2a_an = super_ag.add_node(c2a_an);
                            let c2a_cn = super_cg.add_node(c2a_cn);
                            let (arg_cg_idx, cg_from_outs) = arg.cg_idx.unwrap();
                            if cg_from_outs.len() != 1 {
                                return Err(miette!("Parsing error (graph `{}`): expected expression to have a single output (has {})", &graph_id.0, cg_from_outs.len()));
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
                                    sink_input: audio_arg_count,
                                },
                            );
                            audio_arg_count += 1;

                            continue 'arg_iter;
                        }
                        (ParsedSignalRate::Audio, ParsedSignalRate::Control) => {
                            // insert auto-a2c
                            let (a2c_an, a2c_cn) = AudioToControl::create_nodes(&format!(
                                "auto_a2c_{:?}_{}",
                                callee, arg_id
                            ));
                            let a2c_an = super_ag.add_node(a2c_an);
                            let a2c_cn = super_cg.add_node(a2c_cn);
                            let (arg_ag_idx, ag_from_outs) = arg.ag_idx.unwrap();
                            if ag_from_outs.len() != 1 {
                                return Err(miette!("Parsing error (graph `{}`): expected expression to have a single output (has {})", &graph_id.0, ag_from_outs.len()));
                            }
                            super_ag.add_edge(
                                arg_ag_idx,
                                a2c_an,
                                Connection {
                                    source_output: ag_from_outs[0],
                                    sink_input: 0,
                                },
                            );
                            super_cg.add_edge(
                                a2c_cn,
                                called_cg_idx,
                                Connection {
                                    source_output: 0,
                                    sink_input: control_arg_count,
                                },
                            );

                            control_arg_count += 1;

                            continue 'arg_iter;
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
                        return Err(miette!("Parsing error (graph `{}`): expected expression to have a single output (has {})", &graph_id.0, ag_from_outs.len()));
                    }
                    super_ag.add_edge(
                        ag_from_idx,
                        called_ag_idx,
                        Connection {
                            source_output: 0,
                            sink_input: audio_arg_count,
                        },
                    );
                    audio_arg_count += 1;
                }
                if let Some((cg_from_idx, cg_from_outs)) = arg.cg_idx {
                    if cg_from_outs.len() != 1 {
                        return Err(miette!("Parsing error (graph `{}`): expected expression to have a single output (has {})", &graph_id.0, cg_from_outs.len()));
                    }
                    super_cg.add_edge(
                        cg_from_idx,
                        called_cg_idx,
                        Connection {
                            source_output: 0,
                            sink_input: control_arg_count,
                        },
                    );

                    control_arg_count += 1;
                }
            }

            let known_rate = known_rate.ok_or_else(|| {
                miette!(
                    "Parsing error (graph `{}`): couldn't determine signal rate of expr",
                    &graph_id.0
                )
            })?;

            let (ag_idx, cg_idx) = match known_rate {
                ParsedSignalRate::Audio => {
                    // todo: remove called control node from supergraph
                    (Some((called_ag_idx, (0..n_outs).collect())), None)
                }
                ParsedSignalRate::Control => {
                    // todo: remove called audio node from supergraph
                    (None, Some((called_cg_idx, (0..n_outs).collect())))
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
    known_graphs: &mut BTreeMap<String, ParsedGraph>,
) -> Result<(Graph, Graph)> {
    let ParsedGraph {
        signature: ParsedSignature { inputs, outputs },
        statements,
        id,
    } = graph;

    let mut audio_inputs = vec![];
    let mut audio_outputs = vec![];
    let mut control_inputs = vec![];
    let mut control_outputs = vec![];

    for inp in inputs.iter() {
        let input = Input {
            name: inp.id.0.to_owned(),
            minimum: inp.minimum.map(Signal::Scalar),
            maximum: inp.maximum.map(Signal::Scalar),
            default: inp.default.map(Signal::Scalar),
            is_ui: graph.id.0 == "Main" && inp.id.1 == Some(ParsedSignalRate::Control), // FIXME: hacky
            implicit: false,
        };
        match inp.id.1 {
            Some(ParsedSignalRate::Audio) => audio_inputs.push(input),
            Some(ParsedSignalRate::Control) => control_inputs.push(input),
            None => {
                // audio_inputs.push(input.clone());
                // control_inputs.push(input);
                match graph.id.1 {
                    Some(ParsedSignalRate::Audio) => audio_inputs.push(input),
                    Some(ParsedSignalRate::Control) => control_inputs.push(input),
                    None => {
                        return Err(miette!(
                            "Parsing error (graph `{}`): couldn't determine signal rate of input `{}`",
                            &id.0, &inp.id.0
                        ));
                    }
                }
            }
        }
    }

    for out in outputs.iter() {
        let output = Output {
            name: out.0.to_owned(),
        };
        match out.1 {
            Some(ParsedSignalRate::Audio) => audio_outputs.push(output),
            Some(ParsedSignalRate::Control) => control_outputs.push(output),
            None => match graph.id.1 {
                Some(ParsedSignalRate::Audio) => audio_outputs.push(output),
                Some(ParsedSignalRate::Control) => control_outputs.push(output),
                None => {
                    return Err(miette!(
                        "Parsing error (graph `{}`): couldn't determine signal rate of output `{}`",
                        &id.0,
                        &out.0
                    ));
                }
            },
        }
    }

    let mut ag = Graph::new(Some(NodeName::new(&id.0)), audio_inputs, audio_outputs);
    let mut cg = Graph::new(Some(NodeName::new(&id.0)), control_inputs, control_outputs);

    for stmt in statements {
        match stmt {
            ParsedStatement::Let(stmt) => {
                let mut lhs_nodes = vec![];
                for to in stmt.lhs.iter() {
                    let rate =
                        to.1.or(graph.id.1)
                            .ok_or_else(|| miette!("Parsing error: (graph `{}`): in `let` statement, couldn't determine rate of lhs ident `{}`", &graph.id.0, &to.0))?;
                    let node = LetBinding::create_node(&to.0);
                    let idx = match rate {
                        ParsedSignalRate::Audio => ag.add_node(node),
                        ParsedSignalRate::Control => cg.add_node(node),
                    };
                    lhs_nodes.push(idx);
                }
                let lhs_ident = if stmt.lhs.len() == 1 {
                    Some(&stmt.lhs[0])
                } else {
                    None
                };
                let rhs = solve_expr(id, &stmt.rhs, &mut ag, &mut cg, known_graphs, lhs_ident)?;
                assert!(
                    rhs.ag_idx.is_some() ^ rhs.cg_idx.is_some(),
                    "todo: exprs using both graphs"
                );

                if let Some((ag_from_idx, ag_from_outs)) = rhs.ag_idx {
                    if ag_from_outs.len() != stmt.lhs.len() {
                        return Err(miette!("Parsing error (graph `{}`): in `let` statement, expected as many left-hand terms as the right-hand expression has outputs ({} vs {})", &id.0, stmt.lhs.len(), ag_from_outs.len()));
                    }
                    for (to_idx, ag_from_out) in lhs_nodes.iter().zip(ag_from_outs.iter()) {
                        // todo: don't assume it's the first input, probably
                        ag.add_edge(
                            ag_from_idx,
                            *to_idx,
                            Connection {
                                source_output: *ag_from_out,
                                sink_input: 0,
                            },
                        );
                    }
                }
                if let Some((cg_from_idx, cg_from_outs)) = rhs.cg_idx {
                    if cg_from_outs.len() != stmt.lhs.len() {
                        return Err(miette!("Parsing error (graph `{}`): in `let` statement, expected as many left-hand terms as the right-hand expression has outputs ({} vs {})", &id.0, stmt.lhs.len(), cg_from_outs.len()));
                    }
                    for (to_idx, cg_from_out) in lhs_nodes.iter().zip(cg_from_outs.iter()) {
                        // todo: don't assume it's the first input, probably
                        cg.add_edge(
                            cg_from_idx,
                            *to_idx,
                            Connection {
                                source_output: *cg_from_out,
                                sink_input: 0,
                            },
                        );
                    }
                }
            }
            ParsedStatement::Connection(conn) => {
                let rhs = solve_expr(id, &conn.rhs, &mut ag, &mut cg, known_graphs, None)?;
                assert!(
                    !(rhs.ag_idx.is_some() && rhs.cg_idx.is_some()),
                    "todo: exprs using both graphs"
                );
                if let Some((ag_from_idx, ag_from_outs)) = rhs.ag_idx {
                    if ag_from_outs.len() != conn.lhs.len() {
                        return Err(miette!("Parsing error (graph `{}`): expected as many left-hand terms as the right-hand expression has outputs ({} vs {})", &id.0, conn.lhs.len(), ag_from_outs.len()));
                    }
                    for (to, ag_from_out) in conn.lhs.iter().zip(ag_from_outs.iter()) {
                        let to_idx = ag.node_id_by_name(&to.0)?;
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
                        return Err(miette!("Parsing error (graph `{}`): expected as many left-hand terms as the right-hand expression has outputs ({} vs {})", &id.0, conn.lhs.len(), cg_from_outs.len()));
                    }
                    for (to, cg_from_out) in conn.lhs.iter().zip(cg_from_outs.iter()) {
                        let to_idx = cg.node_id_by_name(&to.0)?;
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

    ag.write_dot(&format!("{}_audio.dot", &id.0))?;
    cg.write_dot(&format!("{}_control.dot", &id.0))?;

    Ok((ag, cg))
}
