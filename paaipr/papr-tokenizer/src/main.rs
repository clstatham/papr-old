use clap::Parser;
use nom::{
    combinator::recognize,
    complete::take,
    error::ParseError,
    multi::many1_count,
    sequence::{preceded, tuple},
    IResult,
};
use papr_lib::parser2::*;
use std::{env::args, error::Error, fs::File, io::Read, path::PathBuf};

#[derive(clap::Parser)]
struct Args {
    file: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut input = String::new();
    let mut reward = 0isize;
    let args = Args::parse();
    let mut f = File::open(args.file).unwrap();
    f.read_to_string(&mut input).unwrap();
    reward += input
        .split_ascii_whitespace()
        .filter(|i| {
            if let Ok((_, tok)) = Token::parse(i) {
                tok != Token::Whitespace
            } else {
                false
            }
        })
        .count() as isize;
    match Token::many1(&input).map_err(|e| e.to_owned()) {
        Ok((garbage, tokens)) => {
            reward -= garbage.split_ascii_whitespace().count() as isize * 10;

            let tokens = Tokens { tokens: &tokens };
            match many1_count(statement)(tokens) {
                Ok((_garbage, count)) => reward += count as isize * 1000,
                Err(e) => {
                    dbg!(e);
                }
            };
        }
        Err(_e) => {
            // eprintln!("Error: {}", e);
            // reward = -100;
        }
    }
    // reward -= 129;
    println!("{}", reward);
    Ok(())
}
