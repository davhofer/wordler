use crate::{compute_feedback, Feedback, Guess, GuessResult, Guesser, WORDS};
use std::{collections::HashMap, f64};
use hashbrown::HashSet;
use rayon::prelude::*;

// TODO: we currently ONLY guess based on max entropy. however, we should also incorporate the
// probability of a word being the correct solution. while it is uniform, it grows with fewer total
// possible solution words
// Question: how do we do the weighting?

pub struct MaxEntropyGuesser {
    possible_solutions: HashSet<&'static str>,
    initial_guess: Option<&'static str>,
}

impl MaxEntropyGuesser {
    pub fn new() -> Self {
        let possible_solutions: HashSet<&str> = WORDS.split_whitespace().collect();
        Self { possible_solutions, initial_guess: None }
    }

    pub fn set_initial_guess(&mut self, guess: &'static str) {
        self.initial_guess = Some(guess);
    }

    fn update_possible_solutions(&mut self, result: &GuessResult) {
        self.possible_solutions = self.possible_solutions.iter().filter_map(|word| {
            // if word would give the same result mask as the guess, keep it, otherwise drop it
            if compute_feedback(word, &result.guess) == result.feedback {
                Some(*word)
            } else {
                None
            }
        }).collect();
    }
}

impl Guesser for MaxEntropyGuesser {
    fn guess(&mut self, wordlist: &HashSet<&str>, prior_guesses: &Vec<GuessResult>) -> Guess {


        // update possible solution based on the previous guess
        if let Some(guess) = prior_guesses.last() {
            let prev_entropy = (self.possible_solutions.len() as f64).log2();
            self.update_possible_solutions(&guess);
            let last_guess_info = prev_entropy - (self.possible_solutions.len() as f64).log2();

            println!("Result pattern: {:?}", guess.feedback);
            println!("Actual information gain: {last_guess_info}\n");
        }

        println!("Guess #{}", prior_guesses.len()+1);

        let total_entropy = (self.possible_solutions.len() as f64).log2();
        println!("Possible solutions: {}, entropy: {}", self.possible_solutions.len(), total_entropy);


        let uniform_solution_proba = 1.0/(self.possible_solutions.len() as f64);

        if prior_guesses.len() == 0 && let Some(guess_str) = self.initial_guess {
            // tares (entropy: 6.159376455792673)
            // Best first/1-round information gain
            // TODO: replace with best 2/3-round information gain word?
            println!("Using initial guess: {guess_str}");
            return Guess{ 
                guess: String::from(guess_str), 
                entropy: 0.0, 
                solution_probability: 0.0,
            };
        }

        // parallelize the entropy calculation for all guesses
        let best_guess = wordlist.par_iter().map(|guess_str| {
 
            // println!("Checking guess '{guess}'");

            let mut pattern_counts: HashMap<[Feedback; 5], f64> = HashMap::new();
            let mut n: f64 = 0.0;

            for solution in &self.possible_solutions {
                *pattern_counts
                    .entry(compute_feedback(solution, guess_str))
                    .or_insert(0.0) += 1.0; 
                n += 1.0;
            } 
            let entropy = -pattern_counts.iter().map(|(_, count)| {
                let p = *count/n;
                p * p.log2()
            }).sum::<f64>();

            // println!("Entropy: {entropy}");
            Guess {
                guess: guess_str.to_string(),
                entropy,
                solution_probability: if self.possible_solutions.contains(guess_str) { uniform_solution_proba } else { 0.0 }
            }
        }).min_by(|a, b| b.quality().total_cmp(&a.quality())).unwrap();

        println!("Guessing: {best_guess:?}");
        best_guess
    }
}


