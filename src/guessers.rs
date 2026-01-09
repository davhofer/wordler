use crate::{compute_feedback, Feedback, Guess, GuessResult, Guesser, WORDS};
use std::{f64, time::{Duration, Instant}};
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

// TODO: we currently ONLY guess based on max entropy. however, we should also incorporate the
// probability of a word being the correct solution. while it is uniform, it grows with fewer total
// possible solution words
// Question: how do we do the weighting?

fn pattern_partitions(guess: &'static str, possible_solutions: &HashSet<&'static str>) -> HashMap<[Feedback; 5], HashSet<&'static str>> {
    let mut pattern_buckets: HashMap<[Feedback; 5], HashSet<&'static str>> = HashMap::new();
    
    // for each potential solution, compute corresponding feedback pattern
    for solution in possible_solutions {
        pattern_buckets
            .entry(compute_feedback(solution, guess))
            .or_insert(HashSet::new()).insert(solution); 
    } 
    pattern_buckets
}

pub struct MaxEntropyGuesser {
    possible_solutions: HashSet<&'static str>,
    initial_guess: Option<&'static str>,
    verbose: bool,
}

impl MaxEntropyGuesser {
    pub fn new() -> Self {
        let possible_solutions: HashSet<&str> = WORDS.split_whitespace().collect();
        Self { possible_solutions, initial_guess: None, verbose: false, }
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

    pub fn compute_best_guess(&self, wordlist: &HashSet<&str>) -> Guess {

        // parallelize the entropy calculation for all guesses
        wordlist.par_iter().map(|guess_str| {
 
            // println!("Checking guess '{guess}'");

            let mut pattern_counts: HashMap<[Feedback; 5], f64> = HashMap::new();
            
            let num_solutions = self.possible_solutions.len() as f64;
            let uniform_solution_proba = 1.0/num_solutions;

            for solution in &self.possible_solutions {
                *pattern_counts
                    .entry(compute_feedback(solution, guess_str))
                    .or_insert(0.0) += 1.0; 
            } 
            let entropy = -pattern_counts.iter().map(|(_, count)| {
                let p = *count/num_solutions;
                p * p.log2()
            }).sum::<f64>();

            // println!("Entropy: {entropy}");
            Guess {
                guess: guess_str.to_string(),
                entropy,
                solution_probability: if self.possible_solutions.contains(guess_str) { uniform_solution_proba } else { 0.0 }
            }
        }).min_by(|a, b| b.quality().total_cmp(&a.quality())).unwrap()
    }

    pub fn compute_best_initial_guess(&self, wordlist: &HashSet<&'static str>) -> (f64, Guess) {
        // computes the guess with the largest expected information gain after 2 rounds
        
        // NOTE: we only use max entropy here 

        // go over all guesses to compute their level 1 entropy and subsequent expected l2 entropy
        wordlist.par_iter().map(|guess_str| {
 
            let mut pattern_buckets: HashMap<[Feedback; 5], HashSet<&'static str>> = pattern_partitions(guess_str, &self.possible_solutions);
            
            let num_solutions = self.possible_solutions.len() as f64;
            let uniform_solution_proba = 1.0/num_solutions;

            // for each potential solution, compute corresponding feedback pattern
            for solution in &self.possible_solutions {
                pattern_buckets
                    .entry(compute_feedback(solution, guess_str))
                    .or_insert(HashSet::new()).insert(solution); 
            } 

            

            // level 2 entropy analysis

            // TODO: does par_iter make sense here? or obsolete as compute_best_guess also will do
            // par iter?
            let (l1_entropy, expected_l2_entropy) = pattern_buckets.par_iter().map(|(_, bucket)| {
                // compute probability of getting this pattern bucket as the correct one
                let bucket_proba = (bucket.len() as f64)/num_solutions;

                // compute best l2 guess for bucket, i.e. expected information gain in the next
                // step

                let best_l2_guess = self.compute_best_guess(bucket);
                
                // expected level 2 entropy := "weighted sum of all L2 entropies" = Sum_(pattern) {
                // p(pattern) * best_l2_guess(pattern).entropy }
                let expected_l2_entropy = bucket_proba * best_l2_guess.entropy;
                // level 1 entropy = - Sum_(pattern) { p(pattern) * log2( p(pattern) ) }
                let l1_entropy = bucket_proba * bucket_proba.log2();
                (l1_entropy, expected_l2_entropy)
            }).reduce(
                    || (0.0, 0.0), 
                    |(a_l1_ent, a_l2_ent), (b_l1_ent, b_l2_ent)| 
                    (a_l1_ent+b_l1_ent, a_l2_ent+b_l2_ent));

            // TODO: could achieve the same using (a_l1_ent - b_l1_ent) above, right?
            let l1_entropy = -l1_entropy;

            let score = l1_entropy + expected_l2_entropy;

            println!("ok");
            // return combined entropy score & Guess
            (score, Guess {
                guess: guess_str.to_string(),
                entropy: l1_entropy,
                solution_probability: if self.possible_solutions.contains(guess_str) { 
                    uniform_solution_proba 
                } else { 
                    0.0 
                }
            })
        }).min_by(|a, b| b.0.total_cmp(&a.0)).unwrap()
    }
}

impl Guesser for MaxEntropyGuesser {
    fn guess(&mut self, wordlist: &HashSet<&str>, prior_guesses: &Vec<GuessResult>) -> Guess {


        // update possible solution based on the previous guess
        if let Some(guess) = prior_guesses.last() {
            let prev_entropy = (self.possible_solutions.len() as f64).log2();
            self.update_possible_solutions(&guess);
            let last_guess_info = prev_entropy - (self.possible_solutions.len() as f64).log2();

            if self.verbose {
                println!("Result pattern: {:?}", guess.feedback);
                println!("Actual information gain: {last_guess_info}\n");
            }
        }


        let total_entropy = (self.possible_solutions.len() as f64).log2();

        if self.verbose {
            println!("Guess #{}", prior_guesses.len()+1);
            println!("Possible solutions: {}, entropy: {}", self.possible_solutions.len(), total_entropy);
        }

        if prior_guesses.len() == 0 && let Some(guess_str) = self.initial_guess {
            // tares (entropy: 6.159376455792673)
            // Best first/1-round information gain
            // TODO: replace with best 2/3-round information gain word?
            if self.verbose {
                println!("Using initial guess: {guess_str}");
            }
            return Guess{ 
                guess: String::from(guess_str), 
                entropy: 0.0, 
                solution_probability: 0.0,
            };
        }

        let best_guess = self.compute_best_guess(wordlist);

        if self.verbose {
            println!("Guessing: {best_guess:?}");
        }
        best_guess
    }
}

pub struct MinExpectedScoreGuesser {
    wordlist: HashSet<&'static str>,
    initial_guess: Option<&'static str>,
    verbose: bool,
}

impl MinExpectedScoreGuesser {
    pub fn new() -> Self {
        let wordlist: HashSet<&str> = WORDS.split_whitespace().collect();
        Self { wordlist, initial_guess: None, verbose: false, }
    }

    // TODO: recursion, does this work? memory consumption? make more efficient?
    // bottom up instead of top down?
    pub fn expected_score(&self, guess: &'static str, possible_solutions: &HashSet<&'static str>) -> f64 {
        let p_guess = if possible_solutions.contains(guess) {
            1.0/(possible_solutions.len() as f64) 
        } else {
            0.0
        };

        // compute pattern buckets with corresponding possible solutions
        let buckets = pattern_partitions(guess, possible_solutions);

        // compute score and size for each bucket
        // score is the min expected score across all possible guesses
        let bucket_score_size = buckets.par_iter()
            .map(|(_, solution_set)| {
                (
                    self.wordlist.par_iter().map(|possible_guess| { 
                        self.expected_score(possible_guess, solution_set)
                    }).min_by(|a, b| b.total_cmp(&a)).unwrap(), 
                    solution_set.len()
                )
            });

        // compute weighted avg of scores of all buckets
        let weighted_avg_score: f64 = bucket_score_size
            .map(|(score, size)| score * (size as f64)/(possible_solutions.len() as f64))
            .sum();

        // expected score is 1 if solution is guess, and weighted_avg_score otherwise
        p_guess * 1.0 + (1.0 - p_guess) * weighted_avg_score 
        // TODO: lazily compute weighted_avg_score?
    }

}

//  Basic idea: directly compute expected score of a guess given a set of possible solutions.
//  to do this, use recursive approach. 
//  do pattern partition, and for each bucket, pick the guess that minimizes the expected score for
//  that bucket using same function. then average all these using the bucket probabilities
//  base case is expected guesses = 1 if one solution is left
//  TODO: how to compute expected score before that? because e.g. if we have few options left, it
//  might make more sense to guess one of the options directly?
//  => TODO: work out the math of expected score!!
//  => need to incorporate probability of guess being the right solution into the formula!
//
//
//  E_score[guess] = {
//      buckets = pattern_partitions(guess, possible_solutions)
//      for each bucket, bucket_guess = min_(word)(E_score[word]) // take guess for bucket that
//      minimizes the expected score if solution is in that bucket
//
//      e_score = 1 * proba("guess is solution") + [ weighted average of expected scores for the
//      different buckets (?) ] * (1- proba("guess is solution"))
//      => in this second case, we can assume that guess is not the solution, so remove it from
//      possible solutions if it's in there
//  }
//  proba("guess is solution") is just uniform over possible_solutions (and 0 if guess is not in
//  possible_solutions)
//
//
//  think about pattern_parititons function. can we also use it for MaxEntropyGuesser?
//  
//  think about complexity/runtime of this code
//
//  memoization?
//
//
//
//
//  can we shortcut this computation?
//  is expected number of guesses simply dependent on bucket size? => no, pattern matters

// TODO: make the same for all guessers, how wordlist and possible solutions are stored/handled?
// struct field? argument?

// TODO: do we need to change the Guess struct? entropy does not make sense for other types of
// guessers, so maybe change abstraction?
impl Guesser for MinExpectedScoreGuesser {
    fn guess(&mut self, wordlist: &HashSet<&str>, prior_guesses: &Vec<GuessResult>) -> Guess {



        Guess { guess: String::new(), entropy: 0.0, solution_probability: 0.0 }    
    }
}


