use crate::{Guess, GuessResult, compute_feedback, WORDS, Feedback};
use std::{collections::{HashMap, HashSet}, f64};
use rand::{rng, seq::IteratorRandom};

pub struct RandomGuesser {}

impl Guess for RandomGuesser {
    fn guess(&mut self, wordlist: &HashSet<&str>, _prior_guesses: &Vec<GuessResult>) -> String {
        wordlist.iter().choose(&mut rng()).map(|word| *word).unwrap_or("oomph").to_string()
    }
}


pub struct MaxEntropyGuesser {
    pub possible_solutions: HashSet<&'static str>,
}

impl MaxEntropyGuesser {
    pub fn new() -> Self {
        let possible_solutions: HashSet<&str> = WORDS.split_whitespace().collect();
        Self { possible_solutions }
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

// TODO: we currently ONLY guess based on max entropy. however, we should also incorporate the
// probability of a word being the correct solution. while it is uniform, it grows with fewer total
// possible solution words
// Question: how do we do the weighting?

impl Guess for MaxEntropyGuesser {
    fn guess(&mut self, wordlist: &HashSet<&str>, prior_guesses: &Vec<GuessResult>) -> String {


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

        if self.possible_solutions.len() == 1 {
            let result = self.possible_solutions.iter().last().unwrap().to_string();
            println!("Guessing: {result}");
            return result;
        }


        // HARDCODED FIRST GUESS
        if prior_guesses.len() == 0 {
            println!("Guessing: crate (fixed first guess)");
            return String::from("crate");
        }

        // compute entropy (expected information gain) for all possible guesses
        //  to compute it, go through all possible solution words, compute the resulting patterns,
        //  compute the probability of each pattern, and the subsequent expected information gain
        //  Info(event) = -log2(Prob(event))
        //  an event here would be observing a specific pattern 
        //  to compute the proba of observing a specific pattern, simply count for how many
        //  solution words this pattern would occur
        //  this is how we get 
        //  E[I] = H(p) = -Sum_x[p(x) * log2(p(x))]
        //  NOTE: p(x) here is not the proba of a word being the solution, it's the proba of a
        //  specific pattern occuring


        // for each word in wordlist:
        //   
        //   for each possible solution:
        //     
        //     compute result pattern 
        //     increase count in HashSet
        //     increase counter
        // 
        //   compute probas and entropy for word (guess) based on pattern counts
        // choose best guess based on max entropy

        let mut best_guess = "";
        let mut best_score = f64::MIN;
        for guess in wordlist {

            // println!("Checking guess '{guess}'");

            let mut pattern_counts: HashMap<[Feedback; 5], f64> = HashMap::new();
            let mut n: f64 = 0.0;

            for solution in &self.possible_solutions {
                *pattern_counts
                    .entry(compute_feedback(solution, *guess))
                    .or_insert(0.0) += 1.0; 
                n += 1.0;
            } 
            let entropy = -pattern_counts.iter().map(|(_, count)| {
                let p = *count/n;
                p * p.log2()
            }).sum::<f64>();

            // println!("Entropy: {entropy}");

            if entropy > best_score {
                best_score = entropy;
                best_guess = *guess;
            }
        } 
        println!("Guessing: {best_guess}\n(Entropy: {best_score})");
        best_guess.to_string()
    }
}
