use super::*;

macro_rules! test_parse {
    ($tester:ident, $rule:ident, $input:expr) => {
        $tester(
            $input,
            PaprParser::parse(Rule::$rule, $input)
                .map_err(|e| ParseError::from_source_and_pest_error($input, e))?
                .next()
                .unwrap(),
        )?
    };
}

macro_rules! test_parse_fail {
    ($rule:ident, $input:expr) => {
        PaprParser::parse(Rule::$rule, $input).unwrap_err()
    };
}

#[test]
fn test_parse_scalar() -> Result<()> {
    test_parse!(parse_scalar, scalar, "1234567890");
    test_parse!(parse_scalar, scalar, "1234567890.1234567890");
    Ok(())
}

#[test]
fn test_parse_string() -> Result<()> {
    assert_eq!(test_parse!(parse_string, string, "\"123abc\""), "123abc");
    test_parse_fail!(string, "123abc");
    Ok(())
}

#[test]
fn test_parse_ident() -> Result<()> {
    test_parse!(parse_ident, ident, "foobar");
    test_parse!(parse_ident, ident, "@foobar");
    test_parse!(parse_ident, ident, "#fooBar");
    Ok(())
}

#[test]
fn test_parse_graph_name() -> Result<()> {
    test_parse!(parse_graph_name, graph_name, "Foobar");
    test_parse!(parse_graph_name, graph_name, "@Foobar");
    test_parse!(parse_graph_name, graph_name, "#FooBar");
    test_parse_fail!(graph_name, "foobar");
    test_parse_fail!(graph_name, "@foobar");
    test_parse_fail!(graph_name, "#fooBar");
    Ok(())
}

#[test]
fn test_parse_import() -> Result<()> {
    assert_eq!(
        test_parse!(parse_import, import_stmt, "import \"foobar\""),
        "foobar"
    );
    test_parse_fail!(import_stmt, "import ");
    test_parse_fail!(import_stmt, "\"foobar\"");
    Ok(())
}

#[test]
fn test_parse_call() -> Result<()> {
    test_parse!(parse_call, call, "Foo()");
    test_parse!(parse_call, call, "Foo(1 2 3)");
    test_parse!(parse_call, call, "Foo<4 5 6>(1 2 3)");
    test_parse!(parse_call, call, "#Foo<4 5 6>(1 2 3)");
    test_parse!(parse_call, call, "#Foo<4 \"string\" 6>(1 2 3)");
    test_parse!(parse_call, call, "#Foo<4 \"string\" 6>(@Bar(7 8 9) 2 3)");
    test_parse!(
        parse_call,
        call,
        "#Foo<4 \"string\" 6>(@Bar(7 8 9) 2 (10 * 11))"
    );
    Ok(())
}

#[test]
fn test_connection() -> Result<()> {
    test_parse!(parse_connection, connection, "#foo = #Bar(1);");
    Ok(())
}

#[test]
fn test_statement() -> Result<()> {
    test_parse!(parse_statement, statement, "#foo = #Bar(1);");
    test_parse!(parse_statement, statement, "let #foo = #Bar(1);");
    Ok(())
}
