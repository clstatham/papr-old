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

macro_rules! test_parse_many {
   ($($test_name:ident => $tester:expr,)*) => {
        $(
            #[test]
            fn $test_name() -> Result<()> {
                $tester;
                Ok(())
            }
        )*
    };
}

macro_rules! test_parse_fail {
    ($rule:ident, $input:expr) => {
        PaprParser::parse(Rule::$rule, $input).unwrap_err()
    };
}

test_parse_many! {
    test_parse_scalar_1 => test_parse!(parse_scalar, scalar, "1234567890"),
    test_parse_scalar_2 => test_parse!(parse_scalar, scalar, "1234567890.1234567890"),

    test_parse_string_1 => assert_eq!(test_parse!(parse_string, string, "\"123abc\""), "123abc"),

    test_parse_ident_1 => test_parse!(parse_ident, ident, "foobar"),
    test_parse_ident_2 => test_parse!(parse_ident, ident, "@foobar"),
    test_parse_ident_3 => test_parse!(parse_ident, ident, "#fooBar"),

    test_parse_graph_name_1 => test_parse!(parse_graph_name, graph_name, "Foobar"),
    test_parse_graph_name_2 => test_parse!(parse_graph_name, graph_name, "@Foobar"),
    test_parse_graph_name_3 => test_parse!(parse_graph_name, graph_name, "#FooBar"),

    test_parse_import_1 => assert_eq!(test_parse!(parse_import, import_stmt, "import \"foobar\""), "foobar"),

    test_parse_call_1 => test_parse!(parse_call, call, "Foo()"),
    test_parse_call_2 => test_parse!(parse_call, call, "Foo(1, 2, 3)"),
    test_parse_call_3 => test_parse!(parse_call, call, "Foo<4, 5, 6>(1, 2, 3)"),
    test_parse_call_4 => test_parse!(parse_call, call, "#Foo<4, 5, 6>(1, 2, 3)"),
    test_parse_call_5 => test_parse!(parse_call, call, "#Foo<4, \"string\", 6>(1, 2, 3)"),
    test_parse_call_6 => test_parse!(parse_call, call, "#Foo<4, \"string\", 6>(@Bar(7, 8, 9), 2, 3)"),
    test_parse_call_7 => test_parse!(parse_call, call, "#Foo<4, \"string\", 6>(@Bar(7, 8, 9), 2, (10 * 11))"),
    test_parse_call_8 => test_parse!(parse_call, call, "#Foo<4, \"string\", 6>(@Bar(7, 8, 9), 2, 10 * 11)"),

    test_parse_connection_1 => test_parse!(parse_connection, connection, "#foo = #Bar(1);"),

    test_parse_statement_1 => test_parse!(parse_statement, statement, "#foo = #Bar(1);"),
    test_parse_statement_2 => test_parse!(parse_statement, statement, "let #foo = #Bar(1);"),
}
