use crate::{Guess, GuessResult};
use std::collections::HashSet;
use rand::{rng, seq::IteratorRandom};

pub struct RandomGuesser {}

impl Guess for RandomGuesser {
    fn guess(&self, wordlist: &HashSet<&str>, _prior_guesses: &Vec<GuessResult>) -> String {
        wordlist.iter().choose(&mut rng()).map(|word| *word).unwrap_or("oomph").to_string()
    }
}


pub struct MaxEntropyGuesser {}

impl Guess for MaxEntropyGuesser {
    fn guess(&self, wordlist: &HashSet<&str>, prior_guesses: &Vec<GuessResult>) -> String {
        String::from("hello")
    }
}
