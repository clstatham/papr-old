use clap::Parser;
use papr_lib::parser3::*;
use pest::error::Error as PestError;
use pest::Parser as PestParser;
use std::{collections::HashMap, error::Error, fs::File, io::Read, path::PathBuf};

fn gen_map() {
    // let mut out = HashMap::new();

    let mut toks = String::new();
    {
        let mut f = File::open("builtin_tokens.txt").unwrap();
        f.read_to_string(&mut toks).unwrap();
    }
    let builtin_toks = toks.split_ascii_whitespace().collect::<Vec<_>>();

    toks.clear();
    {
        let mut f = File::open("keyword_tokens.txt").unwrap();
        f.read_to_string(&mut toks).unwrap();
    }
    let keyword_toks = toks.split_ascii_whitespace().collect::<Vec<_>>();

    toks.clear();
    {
        let mut f = File::open("operator_tokens.txt").unwrap();
        f.read_to_string(&mut toks).unwrap();
    }
    let operator_toks = toks.split_ascii_whitespace().collect::<Vec<_>>();

    toks.clear();
    {
        let mut f = File::open("var_tokens.txt").unwrap();
        f.read_to_string(&mut toks).unwrap();
    }
    let var_toks = toks.split_ascii_whitespace().collect::<Vec<_>>();
}
#[derive(clap::Parser)]
struct Args {
    file: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut orig_input = String::new();
    let mut reward = 0isize;
    let args = Args::parse();
    let mut f = File::open(args.file).unwrap();
    f.read_to_string(&mut orig_input).unwrap();

    let tokens = orig_input.split_ascii_whitespace().collect::<Vec<_>>();
    let mut last_token = None;
    for i in 0..tokens.len() {
        // if PaprParser::parse(Rule::graph_def, &tokens[i..].join(" ")).is_ok() {
        //     reward += 10;
        //     // eprintln!("found graph");
        // }
        if PaprParser::parse(Rule::statement, &tokens[i..].join(" ")).is_ok() {
            reward += 10;
            // eprintln!("found statement");
        }
        // if PaprParser::parse(Rule::expr, &tokens[i..].join(" ")).is_ok() {
        //     reward += 1;
        // }
        if let Ok(mut id) = PaprParser::parse(Rule::ident, &tokens[i..].join(" ")) {
            if let Some(id_next) = id.next() {
                let id = parse_ident(&tokens[i..].join(" "), id_next);
                if let (Ok(id), Some(last)) = (id.as_ref(), last_token.as_ref()) {
                    if id.0.as_str() == last {
                        reward -= 1;
                        // eprintln!("found repeat");
                    } else {
                        reward += 1;
                    }

                    // if id.0.contains("dac0") {
                    //     reward += 1;
                    // }
                }
                last_token = id.ok().map(|i| i.0.clone());
            }
        }
    }
    // reward -= 10; // offset for the template
    println!("{}", reward);
    Ok(())
}
