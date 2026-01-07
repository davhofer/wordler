use crate::{Guess, GuessResult, compute_feedback, WORDS, Feedback};
use std::{cmp::Ordering, collections::{HashMap, HashSet}, f64, thread};
use rand::{rng, seq::IteratorRandom};

pub struct RandomGuesser {}

impl Guess for RandomGuesser {
    fn guess(&mut self, wordlist: &HashSet<&str>, _prior_guesses: &Vec<GuessResult>) -> String {
        wordlist.iter().choose(&mut rng()).map(|word| *word).unwrap_or("oomph").to_string()
    }
}


pub struct MaxEntropyGuesser {
    possible_solutions: HashSet<&'static str>,
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

        // if one possible solution is left, return it as the final guess
        if self.possible_solutions.len() == 1 {
            let result = self.possible_solutions.iter().last().unwrap().to_string();
            println!("Guessing: {result}");
            return result;
        }

        // HARDCODED FIRST GUESS
        if prior_guesses.len() == 0 {
            // tares (entropy: 6.159376455792673)
            // Best first/1-round information gain
            // TODO: replace with best 2/3-round information gain word?
            println!("Guessing: tares (entropy: 6.159376455792673)");
            return String::from("tares");
        }

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


pub struct ParallelMaxEntropyGuesser {
    possible_solutions: HashSet<&'static str>,
    n_threads: usize,
}

impl ParallelMaxEntropyGuesser {
    pub fn new() -> Self {
        let possible_solutions: HashSet<&str> = WORDS.split_whitespace().collect();
        let n_threads = usize::max(1, thread::available_parallelism().unwrap().get() - 2);
        println!("Using ParallelMaxEntropyGuesser with {n_threads} threads");
        Self { possible_solutions, n_threads }
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

// NOTE: same algorithm as normal MaxEntropyGuesser, but parallelization allows us to actually
// compute the best first guess and its entropy
impl Guess for ParallelMaxEntropyGuesser {
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

        // if one possible solution is left, return it as a guess 
        // required as we havent introduced solution probability weighting yet
        if self.possible_solutions.len() == 1 {
            let result = self.possible_solutions.iter().last().unwrap().to_string();
            println!("Guessing: {result}");
            return result;
        }

        // TODO: can precompute these wordlist chunks
        let refs: Vec<&str> = wordlist.iter().copied().collect();
        let chunk_size = wordlist.len().div_ceil(self.n_threads);

        let solutions = &self.possible_solutions;

        let (best_score, best_guess) = thread::scope(|s| {
            let mut handles = Vec::new();

            for chunk in refs.chunks(chunk_size) {
                handles.push(s.spawn(move || {
                    let mut best_guess = "";
                    let mut best_score = f64::NEG_INFINITY;
                    for &guess in chunk {
                    
                        let mut pattern_counts: HashMap<[Feedback; 5], f64> = HashMap::new();
                        let mut n: f64 = 0.0;

                        for solution in solutions {
                            *pattern_counts
                                .entry(compute_feedback(solution, guess))
                                .or_insert(0.0) += 1.0; 
                            n += 1.0;
                        } 
                        let entropy = -pattern_counts.iter().map(|(_, count)| {
                            let p = *count/n;
                            p * p.log2()
                        }).sum::<f64>();

                        if entropy > best_score {
                            best_score = entropy;
                            best_guess = guess;
                        }
                    }
                    (best_score, best_guess)
                    
                }));
            }

            handles.into_iter()
                .map(|handle| handle.join().unwrap())
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
                .unwrap()


        });

        println!("Guessing: {best_guess} (entropy: {best_score})");
        best_guess.to_string()
    }
}
