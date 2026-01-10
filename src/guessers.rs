use crate::{Feedback, Guess, GuessResult, GuessType, Guesser, WORDS, compute_feedback};
use dashmap::DashMap;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::{f64, time::Instant};

// TODO: we currently ONLY guess based on max entropy. however, we should also incorporate the
// probability of a word being the correct solution. while it is uniform, it grows with fewer total
// possible solution words
// Question: how do we do the weighting?

// TODO: introduce a very slight bias towards more common words, after all.
// for example: manic over amnic

fn pattern_partitions(
    guess: &'static str,
    possible_solutions: &HashSet<&'static str>,
) -> HashMap<[Feedback; 5], HashSet<&'static str>> {
    let mut pattern_buckets: HashMap<[Feedback; 5], HashSet<&'static str>> = HashMap::new();

    // for each potential solution, compute corresponding feedback pattern
    for solution in possible_solutions {
        pattern_buckets
            .entry(compute_feedback(solution, guess))
            .or_insert(HashSet::new())
            .insert(solution);
    }
    pattern_buckets
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct SolutionSetKey {
    indices: Vec<usize>,
}

impl SolutionSetKey {
    fn from_set(
        solutions: &HashSet<&'static str>,
        word_to_idx: &HashMap<&'static str, usize>,
    ) -> Self {
        let mut indices: Vec<usize> = solutions
            .iter()
            .filter_map(|word| word_to_idx.get(word).copied())
            .collect();
        indices.sort_unstable();
        Self { indices }
    }
}

pub struct MaxEntropyGuesser {
    possible_solutions: HashSet<&'static str>,
    wordlist: HashSet<&'static str>,
    initial_guess: Option<&'static str>,
    verbose: bool,
    guesses_made: u32,
    best_guess_cache: DashMap<SolutionSetKey, (Guess, f32)>,
    word_to_idx: HashMap<&'static str, usize>,
}

// TODO: don't just compute the best initial guess, compute top n

impl MaxEntropyGuesser {
    pub fn new() -> Self {
        let wordlist: HashSet<&'static str> = WORDS.split_whitespace().collect();
        let possible_solutions: HashSet<&'static str> = WORDS.split_whitespace().collect();
        let word_to_idx: HashMap<&'static str, usize> = wordlist
            .iter()
            .enumerate()
            .map(|(i, &word)| (word, i))
            .collect();
        Self {
            wordlist,
            possible_solutions,
            initial_guess: None,
            verbose: false,
            guesses_made: 0,
            best_guess_cache: DashMap::new(),
            word_to_idx,
        }
    }

    pub fn use_wordlist_subset(&mut self, size: usize) {
        self.wordlist = WORDS.split_whitespace().take(size).collect();
        self.possible_solutions = self.wordlist.clone();
    }

    pub fn set_verbose(&mut self) {
        self.verbose = true;
    }

    pub fn set_initial_guess(&mut self, guess: &'static str) {
        self.initial_guess = Some(guess);
    }

    fn update_possible_solutions(&mut self, result: &GuessResult) {
        self.possible_solutions = self
            .possible_solutions
            .iter()
            .filter_map(|word| {
                // if word would give the same result mask as the guess, keep it, otherwise drop it
                if compute_feedback(word, &result.guess) == result.feedback {
                    Some(*word)
                } else {
                    None
                }
            })
            .collect();
    }

    // TODO: would we benefit from clearing the cache once in a while? do we risk hash collisions?
    // performance loss?
    pub fn compute_best_guess(&self, possible_solutions: &HashSet<&'static str>) -> Guess {
        let size = possible_solutions.len();
        // base case for performance
        if size == 1 {
            return Guess {
                guess: possible_solutions.iter().next().unwrap(),
                variant: GuessType::Entropy {
                    entropy: 0.0,
                    solution_probability: 1.0,
                },
            };
        }

        let key = SolutionSetKey::from_set(possible_solutions, &self.word_to_idx);
        if size >= 10
            && let Some(cached) = self.best_guess_cache.get(&key)
        {
            let duration = (*cached).1;
            // println!(
            //     "cache hit! saved {duration:.2}s, set size: {}",
            //     possible_solutions.len()
            // );
            return (*cached).0.clone();
        }
        let start = Instant::now();

        // let map_func = |&guess_str| {
        //     let mut pattern_counts: HashMap<[Feedback; 5], f64> = HashMap::new();
        //
        //     let num_solutions = possible_solutions.len() as f64;
        //     let uniform_solution_proba = 1.0 / num_solutions;
        //
        //     for solution in possible_solutions {
        //         *pattern_counts
        //             .entry(compute_feedback(solution, guess_str))
        //             .or_insert(0.0) += 1.0;
        //     }
        //     let entropy = -pattern_counts
        //         .iter()
        //         .map(|(_, count)| {
        //             let p = *count / num_solutions;
        //             p * p.log2()
        //         })
        //         .sum::<f64>();
        //
        //     // println!("Entropy: {entropy}");
        //     Guess {
        //         guess: guess_str,
        //         variant: GuessType::Entropy {
        //             entropy,
        //             solution_probability: if possible_solutions.contains(guess_str) {
        //                 uniform_solution_proba
        //             } else {
        //                 0.0
        //             },
        //         },
        //     }
        // };
        //
        // // only use par_iter for large sets
        // let result = if possible_solutions.len() > 100 {
        //     self.wordlist()
        //         .par_iter()
        //         .map(map_func)
        //         .min_by(|a, b| b.quality().total_cmp(&a.quality()))
        //         .unwrap()
        // } else {
        //     self.wordlist()
        //         .iter()
        //         .map(map_func)
        //         .min_by(|a, b| b.quality().total_cmp(&a.quality()))
        //         .unwrap()
        // };

        let result = self
            .wordlist()
            .par_iter()
            .map(|guess_str| {
                let mut pattern_counts: HashMap<[Feedback; 5], f64> = HashMap::new();

                let num_solutions = possible_solutions.len() as f64;
                let uniform_solution_proba = 1.0 / num_solutions;

                for solution in possible_solutions {
                    *pattern_counts
                        .entry(compute_feedback(solution, guess_str))
                        .or_insert(0.0) += 1.0;
                }
                let entropy = -pattern_counts
                    .iter()
                    .map(|(_, count)| {
                        let p = *count / num_solutions;
                        p * p.log2()
                    })
                    .sum::<f64>();

                // println!("Entropy: {entropy}");
                Guess {
                    guess: guess_str,
                    variant: GuessType::Entropy {
                        entropy,
                        solution_probability: if possible_solutions.contains(guess_str) {
                            uniform_solution_proba
                        } else {
                            0.0
                        },
                    },
                }
            })
            .min_by(|a, b| b.quality().total_cmp(&a.quality()))
            .unwrap();

        if size >= 10 {
            let duration = (Instant::now() - start).as_secs_f32();
            self.best_guess_cache
                .insert(key, (result.clone(), duration));
        }
        result
    }

    pub fn compute_best_initial_guess(&self) -> (f64, Guess) {
        let possible_solutions = self.wordlist();
        // computes the guess with the largest expected information gain after 2 rounds

        // NOTE: we only use max entropy here

        // go over all guesses to compute their level 1 entropy and subsequent expected l2 entropy
        self.wordlist()
            .par_iter()
            .map(|guess_str| {
                let num_solutions = possible_solutions.len() as f64;
                let uniform_solution_proba = 1.0 / num_solutions;

                let pattern_buckets: HashMap<[Feedback; 5], HashSet<&'static str>> =
                    pattern_partitions(guess_str, possible_solutions);

                // level 2 entropy analysis

                // TODO: does par_iter make sense here? or obsolete as compute_best_guess also will do
                // par iter?
                let (l1_entropy, expected_l2_entropy) = pattern_buckets
                    .par_iter()
                    .map(|(_, bucket)| {
                        // compute probability of getting this pattern bucket as the correct one
                        let bucket_proba = (bucket.len() as f64) / num_solutions;

                        // compute best l2 guess for bucket, i.e. expected information gain in the next
                        // step

                        let best_l2_guess = self.compute_best_guess(bucket);

                        // expected level 2 entropy := "weighted sum of all L2 entropies" = Sum_(pattern) {
                        // p(pattern) * best_l2_guess(pattern).entropy }
                        let expected_l2_entropy = bucket_proba * best_l2_guess.unwrap_entropy().0;
                        // level 1 entropy = - Sum_(pattern) { p(pattern) * log2( p(pattern) ) }
                        let l1_entropy = bucket_proba * bucket_proba.log2();
                        (l1_entropy, expected_l2_entropy)
                    })
                    .reduce(
                        || (0.0, 0.0),
                        |(a_l1_ent, a_l2_ent), (b_l1_ent, b_l2_ent)| {
                            (a_l1_ent + b_l1_ent, a_l2_ent + b_l2_ent)
                        },
                    );

                // TODO: could achieve the same using (a_l1_ent - b_l1_ent) above, right?
                let l1_entropy = -l1_entropy;

                let score = l1_entropy + expected_l2_entropy;

                // return combined entropy score & Guess
                (
                    score,
                    Guess {
                        guess: guess_str,
                        variant: GuessType::Entropy {
                            entropy: l1_entropy,
                            solution_probability: if possible_solutions.contains(guess_str) {
                                uniform_solution_proba
                            } else {
                                0.0
                            },
                        },
                    },
                )
            })
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap()
    }
}

impl Guesser for MaxEntropyGuesser {
    fn guesses_made(&self) -> u32 {
        self.guesses_made
    }

    fn wordlist(&self) -> &HashSet<&'static str> {
        &self.wordlist
    }

    fn possible_solutions(&self) -> &HashSet<&'static str> {
        &self.possible_solutions
    }

    fn update_state(&mut self, guess_result: GuessResult) {
        self.guesses_made += 1;

        let prev_entropy = (self.possible_solutions().len() as f64).log2();
        self.update_possible_solutions(&guess_result);

        let last_guess_info = prev_entropy - (self.possible_solutions.len() as f64).log2();

        if self.verbose {
            println!("Result pattern: {:?}", guess_result.feedback);
            println!("Actual information gain: {last_guess_info}\n");
        }
    }

    fn guess(&self) -> Guess {
        let total_entropy = (self.possible_solutions().len() as f64).log2();

        if self.verbose {
            println!("Guess #{}", self.guesses_made() + 1);
            println!(
                "Possible solutions: {}, entropy: {}",
                self.possible_solutions().len(),
                total_entropy
            );
        }

        if self.guesses_made() == 0
            && let Some(guess_str) = self.initial_guess
        {
            // tares (entropy: 6.159376455792673)
            // Best first/1-round information gain
            // TODO: replace with best 2/3-round information gain word?
            if self.verbose {
                println!("Using initial guess: {guess_str}");
            }
            return Guess {
                guess: guess_str,
                variant: GuessType::Entropy {
                    entropy: 0.0,
                    solution_probability: 0.0,
                },
            };
        }

        let start = Instant::now();
        let best_guess = self.compute_best_guess(self.possible_solutions());
        let duration = (Instant::now() - start).as_secs_f32();

        if self.verbose {
            println!("Guessing: {best_guess:?}\ntook {duration:.2}s");
        }
        best_guess
    }
}

pub struct MinExpectedScoreGuesser {
    wordlist: HashSet<&'static str>,
    possible_solutions: HashSet<&'static str>,
    initial_guess: Option<&'static str>,
    verbose: bool,
    guesses_made: u32,
}

impl MinExpectedScoreGuesser {
    pub fn new() -> Self {
        let wordlist: HashSet<&str> = WORDS.split_whitespace().collect();
        let possible_solutions: HashSet<&str> = WORDS.split_whitespace().collect();
        Self {
            wordlist,
            possible_solutions,
            initial_guess: None,
            verbose: false,
            guesses_made: 0,
        }
    }

    pub fn set_verbose(&mut self) {
        self.verbose = true;
    }

    pub fn set_initial_guess(&mut self, guess: &'static str) {
        self.initial_guess = Some(guess);
    }

    fn update_possible_solutions(&mut self, result: &GuessResult) {
        self.possible_solutions = self
            .possible_solutions
            .iter()
            .filter_map(|word| {
                // if word would give the same result mask as the guess, keep it, otherwise drop it
                if compute_feedback(word, &result.guess) == result.feedback {
                    Some(*word)
                } else {
                    None
                }
            })
            .collect();
    }

    // TODO: memoize expected score
    pub fn compute_best_guess(&self, possible_solutions: &HashSet<&'static str>) -> Guess {
        self.wordlist()
            .par_iter()
            .map(|possible_guess| {
                let guess_variant = GuessType::ExpectedScore {
                    score: self.expected_score(possible_guess, possible_solutions),
                };
                Guess {
                    guess: possible_guess,
                    variant: guess_variant,
                }
            })
            .min_by(|a, b| {
                b.unwrap_expected_score()
                    .total_cmp(&a.unwrap_expected_score())
            })
            .unwrap()
    }

    // TODO: recursion, does this work? memory consumption? make more efficient?
    // bottom up instead of top down?
    pub fn expected_score(
        &self,
        guess: &'static str,
        possible_solutions: &HashSet<&'static str>,
    ) -> f64 {
        println!(
            "computing E[score] of guess '{guess}' with len possible solutions {}",
            possible_solutions.len()
        );

        // compute pattern buckets with corresponding possible solutions
        // O(len(possible_solutions))
        let buckets = pattern_partitions(guess, possible_solutions);

        // compute score and size for each bucket
        // score is the min expected score across all possible guesses
        let bucket_score_size = buckets.par_iter().map(|(_, solution_set)| {
            (
                self.compute_best_guess(solution_set)
                    .unwrap_expected_score(),
                solution_set.len(),
            )
        });

        // compute weighted avg of scores of all buckets
        bucket_score_size
            .map(|(score, size)| score * (size as f64) / (possible_solutions.len() as f64))
            .sum()
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
//
//
//
//
// TODO: IS THIS APPROACH BASICALLY JUST "play all games and count"?
// or is it faster?

impl Guesser for MinExpectedScoreGuesser {
    fn possible_solutions(&self) -> &HashSet<&'static str> {
        &self.possible_solutions
    }

    fn wordlist(&self) -> &HashSet<&'static str> {
        &self.wordlist
    }

    fn guesses_made(&self) -> u32 {
        self.guesses_made
    }

    fn update_state(&mut self, guess_result: GuessResult) {
        self.guesses_made += 1;

        self.update_possible_solutions(&guess_result);

        if self.verbose {
            println!("Result pattern: {:?}", guess_result.feedback);
        }
    }

    fn guess(&self) -> Guess {
        if self.guesses_made() == 0
            && let Some(guess_str) = self.initial_guess
        {
            if self.verbose {
                println!("Using initial guess: {guess_str}");
            }
            return Guess {
                guess: guess_str,
                variant: GuessType::ExpectedScore { score: 0.0 },
            };
        }
        self.compute_best_guess(self.possible_solutions())
    }
}
