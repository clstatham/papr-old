use clap::Parser;
use nom::multi::many1_count;
use papr_lib::parser2::{graph, Token, Tokens};
use std::{error::Error, fs::File, io::Read, path::PathBuf};

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
        .filter(|i| Token::parse(i).is_ok())
        .count() as isize;
    match Token::many1(&input).map_err(|e| e.to_owned()) {
        Ok((garbage, tokens)) => {
            reward += tokens.len() as isize * 10;
            reward -= garbage.split_ascii_whitespace().count() as isize;

            let tokens = Tokens { tokens: &tokens };
            if let Ok((_garbage, count)) = many1_count(graph)(tokens) {
                reward += count as isize * 1000
            }
        }
        Err(_e) => {
            // eprintln!("Error: {}", e);
            // reward = -100;
        }
    }
    println!("{}", reward);
    Ok(())
}
