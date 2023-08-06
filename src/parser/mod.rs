use nom::{
    branch::*, bytes::complete::*, character::complete::*, combinator::*, multi::*, sequence::*,
    IResult,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Binding {
    Audio(String),
    Control(String),
}

#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub audio_inputs: Vec<Binding>,
    pub audio_outputs: Vec<Binding>,
    pub control_inputs: Vec<Binding>,
    pub control_outputs: Vec<Binding>,
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

pub fn ignore_comments<'a, O>(
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
    recognize(many1(alt((alphanumeric1, tag("_")))))
}

pub fn audio_binding<'a>() -> impl FnMut(&'a str) -> IResult<&'a str, Binding> {
    map(preceded(tag("@"), ident()), |id| {
        Binding::Audio(id.to_owned())
    })
}

pub fn control_binding<'a>() -> impl FnMut(&'a str) -> IResult<&'a str, Binding> {
    map(preceded(tag("#"), ident()), |id| {
        Binding::Control(id.to_owned())
    })
}

pub fn in_braces<'a, O>(
    parser: impl FnMut(&'a str) -> IResult<&str, O>,
) -> impl FnMut(&'a str) -> IResult<&str, O> {
    map(
        tuple((
            preceded(tuple((tag("{"), space0)), parser),
            recognize(tuple((space0, tag("}")))),
        )),
        |(a, _)| a,
    )
}

pub fn many_in_braces<'a, O>(
    parser: impl FnMut(&'a str) -> IResult<&str, O>,
) -> impl FnMut(&'a str) -> IResult<&str, Vec<O>> {
    in_braces(preceded(
        whitespace0(),
        many1(map(tuple((parser, whitespace0())), |(p, _)| p)),
    ))
}

pub fn node_def<'a>() -> impl FnMut(&'a str) -> IResult<&str, Node> {
    map(
        tuple((
            preceded(ignore_comments(tag("node")), ident()),
            ignore_comments(in_braces(ignore_comments(tuple((
                ignore_comments(tag("@in")),
                many_in_braces(ignore_comments(audio_binding())),
                ignore_comments(tag("@out")),
                many_in_braces(ignore_comments(audio_binding())),
                ignore_comments(tag("#in")),
                many_in_braces(ignore_comments(control_binding())),
                ignore_comments(tag("#out")),
                many_in_braces(ignore_comments(control_binding())),
                // todo: ~ {}
            ))))),
        )),
        |(id, (_, audio_inputs, _, audio_outputs, _, control_inputs, _, control_outputs))| Node {
            name: id.to_owned(),
            audio_inputs,
            audio_outputs,
            control_inputs,
            control_outputs,
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::parser::*;

    #[test]
    fn test_parse() {
        let txt = "node foo {
    @in { @audio_input0 @audio_input1 }
    @out { @audio_output0 @audio_output1 }
    #in { #control_input0 }
    #out { #control_output0 #control_output1 }
}
";

        dbg!(ignore_comments(node_def())(txt));
    }
}
