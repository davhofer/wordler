use crate::{Feedback, FeedbackStorage, Guess, GuessResult, GuessType, Guesser, WORDS};
use dashmap::DashMap;
use fxhash::FxHasher;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::hash::{Hash, Hasher};
use std::{f64, time::Instant};

// TODO: introduce a slight bias towards more common words?
// for example: manic over amnic

/// Custom hash function to compute map keys from &HashSet<&'static str>
fn unordered_set_hash(set: &HashSet<&'static str>) -> u64 {
    let mut accumulator = 0u64;
    for entry in set {
        let addr = entry.as_ptr() as u64;
        let mut hasher = FxHasher::default();
        addr.hash(&mut hasher);
        let hash = hasher.finish();

        accumulator ^= hash;
    }
    accumulator
}

// TODO: For both guessers, once we know the initial guess only depends on the wordlist, so once we
// know it, can store it and load it at runtime after computing it once
// And for MinExpectedScoreGuesser could actually store and load the whole cache...
// TODO: add a "compute_best_initial_guess" function for both which computes and stores this guess.
// then in the normal guesser usage simply load this initial guess, unless explicitly stated
// otherwise
// Tell the user if expensive computation is happening...

pub struct MaxEntropyGuesserBuilder {
    wordlist_subset: Option<usize>,
    feedback_storage: Option<FeedbackStorage>,
    verbose: bool,
    initial_guess: Option<&'static str>,
}

impl MaxEntropyGuesserBuilder {
    pub fn new() -> Self {
        Self {
            wordlist_subset: None,
            feedback_storage: None,
            verbose: false,
            initial_guess: None,
        }
    }

    pub fn wordlist_subset(mut self, size: usize) -> Self {
        self.wordlist_subset = Some(size);
        self
    }

    pub fn precomputed_patterns(mut self, feedback_storage: FeedbackStorage) -> Self {
        self.feedback_storage = Some(feedback_storage);
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn initial_guess(mut self, initial_guess: &'static str) -> Self {
        self.initial_guess = Some(initial_guess);
        self
    }

    pub fn initial_guess_option(mut self, initial_guess: Option<&'static str>) -> Self {
        self.initial_guess = initial_guess;
        self
    }

    pub fn build(self) -> MaxEntropyGuesser {
        let wordlist: HashSet<&'static str> = match self.wordlist_subset {
            Some(size) => WORDS.split_whitespace().take(size).collect(),
            None => WORDS.split_whitespace().collect(),
        };
        let possible_solutions: HashSet<&'static str> = wordlist.clone();

        MaxEntropyGuesser {
            wordlist,
            possible_solutions,
            initial_guess: self.initial_guess,
            verbose: self.verbose,
            guesses_made: 0,
            best_guess_cache: DashMap::new(),
            feedback_storage: self.feedback_storage,
        }
    }
}

pub struct MaxEntropyGuesser {
    possible_solutions: HashSet<&'static str>,
    wordlist: HashSet<&'static str>,
    initial_guess: Option<&'static str>,
    verbose: bool,
    guesses_made: u32,
    best_guess_cache: DashMap<u64, Guess>,
    feedback_storage: Option<FeedbackStorage>,
}

impl MaxEntropyGuesser {
    pub fn builder() -> MaxEntropyGuesserBuilder {
        MaxEntropyGuesserBuilder::new()
    }

    pub fn new() -> Self {
        Self::builder().build()
    }

    fn get_feedback(&self, solution: &'static str, guess: &'static str) -> [Feedback; 5] {
        if let Some(storage) = &self.feedback_storage {
            storage.get(solution, guess).unwrap()
        } else {
            Feedback::compute(solution, guess)
        }
    }

    fn update_possible_solutions(&mut self, result: &GuessResult) {
        self.possible_solutions = self
            .possible_solutions
            .iter()
            .filter_map(|word| {
                // if word would give the same result mask as the guess, keep it, otherwise drop it
                if self.get_feedback(word, &result.guess) == result.feedback {
                    Some(*word)
                } else {
                    None
                }
            })
            .collect();
    }

    fn pattern_partitions(
        &self,
        guess: &'static str,
        possible_solutions: &HashSet<&'static str>,
    ) -> HashMap<[Feedback; 5], HashSet<&'static str>> {
        let mut pattern_buckets: HashMap<[Feedback; 5], HashSet<&'static str>> = HashMap::new();

        // for each potential solution, compute corresponding feedback pattern
        for solution in possible_solutions {
            pattern_buckets
                .entry(self.get_feedback(solution, guess))
                .or_insert(HashSet::new())
                .insert(solution);
        }
        pattern_buckets
    }

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

        let key = unordered_set_hash(possible_solutions);
        if size >= 10
            && let Some(cached) = self.best_guess_cache.get(&key)
        {
            return (*cached).clone();
        }

        // TODO: only use par_iter for large sets?
        let result = self
            .wordlist()
            .par_iter()
            .map(|guess_str| {
                let mut pattern_counts: HashMap<[Feedback; 5], usize> = HashMap::new();

                let num_solutions = possible_solutions.len() as f64;
                let uniform_solution_proba = 1.0 / num_solutions;

                for solution in possible_solutions {
                    *pattern_counts
                        .entry(self.get_feedback(solution, guess_str))
                        .or_insert(0) += 1;
                }
                let entropy = -pattern_counts
                    .iter()
                    .map(|(_, count)| {
                        let p = (*count as f64) / num_solutions;
                        p * p.log2()
                    })
                    .sum::<f64>();

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
            self.best_guess_cache.insert(key, result.clone());
        }
        result
    }

    /// Computes the guess with the largest expected information gain after 2 rounds if l2_entropy
    /// is set and 1 round otherwise
    pub fn compute_best_initial_guess(&self, l2_entropy: bool) -> (f64, Guess) {
        if !l2_entropy {
            return (0.0, self.compute_best_guess(self.wordlist()));
        }
        let possible_solutions = self.wordlist();

        // go over all guesses to compute their level 1 entropy and subsequent expected l2 entropy
        self.wordlist()
            .par_iter()
            .map(|guess_str| {
                let num_solutions = possible_solutions.len() as f64;
                let uniform_solution_proba = 1.0 / num_solutions;

                let pattern_buckets: HashMap<[Feedback; 5], HashSet<&'static str>> =
                    self.pattern_partitions(guess_str, possible_solutions);

                // level 2 entropy analysis

                // TODO: does par_iter make sense here? or obsolete as compute_best_guess also will do
                // par iter?
                // => compare runtime
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

                // Could achieve the same using (a_l1_ent - b_l1_ent) above, but this is
                // conceptually simpler
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

pub struct MinExpectedScoreGuesserBuilder {
    wordlist_subset: Option<usize>,
    feedback_storage: Option<FeedbackStorage>,
    verbose: bool,
    initial_guess: Option<&'static str>,
    top_k: usize,
}

impl MinExpectedScoreGuesserBuilder {
    pub fn new() -> Self {
        Self {
            wordlist_subset: None,
            feedback_storage: None,
            verbose: false,
            initial_guess: None,
            top_k: 20,
        }
    }

    pub fn wordlist_subset(mut self, size: usize) -> Self {
        self.wordlist_subset = Some(size);
        self
    }

    pub fn precomputed_patterns(mut self, feedback_storage: FeedbackStorage) -> Self {
        self.feedback_storage = Some(feedback_storage);
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn initial_guess(mut self, initial_guess: &'static str) -> Self {
        self.initial_guess = Some(initial_guess);
        self
    }

    pub fn initial_guess_option(mut self, initial_guess: Option<&'static str>) -> Self {
        self.initial_guess = initial_guess;
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn build(self) -> MinExpectedScoreGuesser {
        let wordlist: HashSet<&'static str> = match self.wordlist_subset {
            Some(size) => WORDS.split_whitespace().take(size).collect(),
            None => WORDS.split_whitespace().collect(),
        };
        let possible_solutions: HashSet<&'static str> = wordlist.clone();

        MinExpectedScoreGuesser {
            wordlist,
            possible_solutions,
            initial_guess: self.initial_guess,
            verbose: self.verbose,
            guesses_made: 0,
            best_guess_cache: DashMap::new(),
            feedback_storage: self.feedback_storage,
            used_guesses: HashSet::new(), 
            top_k: self.top_k, 
        } 
    }
}

pub struct MinExpectedScoreGuesser {
    wordlist: HashSet<&'static str>,
    possible_solutions: HashSet<&'static str>,
    initial_guess: Option<&'static str>,
    verbose: bool,
    guesses_made: u32,
    /// Cache of the best guesses for a specific set of possible solutions. Key is derived from
    /// &HashSet<&'static str> using fn unordered_set_hash() and value is a sorted vector of best
    /// guesses (multiple for the case where a guess was already used earlier)
    best_guess_cache: DashMap<u64, Vec<Guess>>,
    /// Already guessed words
    used_guesses: HashSet<&'static str>,
    /// top k guesses (by entropy) to consider
    top_k: usize,
    /// Precomputed feedback patterns for all (solution, guess) pairs
    feedback_storage: Option<FeedbackStorage>,
}

impl MinExpectedScoreGuesser {
    pub fn builder() -> MinExpectedScoreGuesserBuilder {
        MinExpectedScoreGuesserBuilder::new()
    }

    pub fn new() -> Self {
        Self::builder().build()
    }
    // pub fn new() -> Self {
    //     let wordlist: HashSet<&str> = WORDS.split_whitespace().take(2000).collect();
    //     let possible_solutions: HashSet<&str> = WORDS.split_whitespace().take(2000).collect();
    //     Self {
    //         wordlist,
    //         possible_solutions,
    //         initial_guess: None,
    //         verbose: false,
    //         guesses_made: 0,
    //         best_guess_cache: DashMap::new(),
    //         used_guesses: HashSet::new(),
    //         top_k: 200,
    //         feedback_storage: None,
    //     }
    // }

    // pub fn set_verbose(&mut self) {
    //     self.verbose = true;
    // }
    //
    // pub fn set_initial_guess(&mut self, guess: &'static str) {
    //     self.initial_guess = Some(guess);
    // }
    //
    // pub fn set_feedback_storage(&mut self, feedback_storage: FeedbackStorage) {
    //     self.feedback_storage = Some(feedback_storage);
    // }

    fn update_possible_solutions(&mut self, result: &GuessResult) {
        self.possible_solutions = self
            .possible_solutions
            .iter()
            .filter_map(|word| {
                // if word would give the same result mask as the guess, keep it, otherwise drop it
                if Feedback::compute(word, &result.guess) == result.feedback {
                    Some(*word)
                } else {
                    None
                }
            })
            .collect();
    }

    fn pattern_partitions_sorted(
        &self,
        guess: &'static str,
        possible_solutions: &HashSet<&'static str>,
    ) -> Vec<([Feedback; 5], HashSet<&'static str>)> {
        // HashMap<[Feedback; 5], HashSet<&'static str>> {
        let mut pattern_buckets: HashMap<[Feedback; 5], HashSet<&'static str>> = HashMap::new();

        // for each potential solution, compute corresponding feedback pattern
        for solution in possible_solutions {
            pattern_buckets
                .entry(self.get_feedback(solution, guess))
                .or_insert(HashSet::new())
                .insert(solution);
        }

        // pattern_buckets
        let mut buckets_vec: Vec<_> = pattern_buckets.into_iter().collect();
        buckets_vec.sort_by(|a, b| b.1.len().cmp(&a.1.len())); // largest first
        buckets_vec
    }

    fn get_feedback(&self, solution: &'static str, guess: &'static str) -> [Feedback; 5] {
        if let Some(storage) = &self.feedback_storage {
            storage.get(solution, guess).unwrap()
        } else {
            Feedback::compute(solution, guess)
        }
    }

    // TODO: double check if the expected score calculations are correct, with the +1 or not...

    pub fn compute_best_guess(
        &self,
        used_guesses: &HashSet<&'static str>,
        possible_solutions: &HashSet<&'static str>,
        is_first: bool,
    ) -> Guess {
        if is_first && self.verbose {
            println!("\ncompute_best_guess\nused_guesses: {used_guesses:?}");
            if possible_solutions.len() < 10 {
                println!(
                    "possible_solutions ({}): {possible_solutions:?}",
                    possible_solutions.len()
                );
            } else {
                println!("possible_solutions ({}): ...", possible_solutions.len());
            }
        }

        if used_guesses.len() >= 8 {
            // TODO: log2 is too conservative?
            // use a lower bound?
            let proxy_score = (possible_solutions.len() as f64).log2();
            return Guess {
                guess: "aaaaa", // dummy word, won't actually be used
                variant: GuessType::ExpectedScore { score: proxy_score },
            };
        }

        // TODO: how else can we distinguish and prune/eliminate the branches we don't want anyway?

        if possible_solutions.is_empty() {
            return Guess {
                guess: "aaaaa", // dummy word, won't actually be used
                variant: GuessType::ExpectedScore {
                    score: f64::INFINITY,
                },
            };
        }
        if possible_solutions.len() == 1 {
            // println!("BASE CASE AFTER: {}", used_guesses.len());
            let guess = *possible_solutions.iter().next().unwrap();
            let variant = GuessType::ExpectedScore { score: 1.0 };
            let ans = Guess { guess, variant };
            return ans;
        }
        if possible_solutions.len() == 2 {
            // println!("BASE CASE AFTER: {}", used_guesses.len());
            let guess = *possible_solutions.iter().next().unwrap();
            let variant = GuessType::ExpectedScore { score: 1.5 };
            let ans = Guess { guess, variant };
            return ans;
        }

        let key = unordered_set_hash(possible_solutions);

        if let Some(val) = self.best_guess_cache.get(&key) {
            // TODO: is this correct? or only approximately? could be improved?
            // vector length goes up to 6+
            // use a sorted data structure?
            for guess in (*val).clone() {
                if !used_guesses.contains(guess.guess) {
                    if is_first {
                        println!("Cache hit!");
                    }
                    return guess.clone();
                }
            }
        }

        let mut pruned = 0;

        let mut best_score = f64::MAX;
        let mut best_guess = "";

        for guess in self.top_k_entropy_guesses(possible_solutions) {
            let possible_guess = guess.guess;
            if used_guesses.contains(possible_guess) {
                // return None;
                continue;
            }

            // Compute expected score of possible_guess

            // compute pattern buckets with corresponding possible solutions
            // O(len(possible_solutions))
            let buckets = self.pattern_partitions_sorted(possible_guess, possible_solutions);

            // TODO: in whcih order to process buckets for better efficiency/pruning?

            // these are the feedback patterns we could see after making this guess, and the
            // follow up possible solution sets
            //
            // we will need to compute the expected score aggregating the best guesses for each
            // bucket. this is where the recursion takes place

            let mut next_used_guesses = used_guesses.clone();
            next_used_guesses.insert(possible_guess);
            // compute score and size for each bucket
            // score is the min expected score across all possible guesses
            //
            // TODO: instead of actually computing the best guess and its score here, can we
            // use a proxy metric?

            // expected score of the guess currently under consideration
            let mut cur_expected_score = 0.0;
            for (pattern, solution_set) in &buckets {
                let branch_probability =
                    (solution_set.len() as f64) / (possible_solutions.len() as f64);
                if *pattern == [Feedback::Correct; 5] {
                    // our guess was correct; we are done
                    cur_expected_score += 1.0 * branch_probability;
                    continue;
                }

                // TODO: is this a good heuristic?
                let perfect_split_score =
                    (1.0 + (solution_set.len() as f64).log(243.0)) * branch_probability;

                if cur_expected_score + perfect_split_score >= best_score {
                    cur_expected_score += perfect_split_score;
                    pruned += 1;
                    break;
                }

                let recursive_score = self
                    .compute_best_guess(&next_used_guesses, solution_set, false)
                    .unwrap_expected_score();
                cur_expected_score += (1.0 + recursive_score) * branch_probability;

                if cur_expected_score >= best_score {
                    break;
                }
            }
            if cur_expected_score >= best_score {
                pruned += 1;
                continue;
            }

            best_score = cur_expected_score;
            best_guess = possible_guess;
        }
        let ans = Guess {
            guess: best_guess,
            variant: GuessType::ExpectedScore { score: best_score },
        };

        // Try to get mutable reference
        if let Some(mut guesses) = self.best_guess_cache.get_mut(&key) {
            guesses.push(ans.clone());
            // smallest first
            guesses.sort_by(|a, b| a.quality().total_cmp(&b.quality()));
            // println!(" sorted best guesses cache: {:?}", guesses);
        } else {
            // Key doesn't exist, insert new vec
            self.best_guess_cache.insert(key, vec![ans.clone()]);
        }
        // self.best_guess_cache.insert(key, ans.clone());
        if is_first {
            println!("pruned: {pruned}");
        }

        return ans;
    }

    fn top_k_entropy_guesses(&self, possible_solutions: &HashSet<&'static str>) -> Vec<Guess> {
        let size = possible_solutions.len();
        // base case for performance
        if size == 1 {
            return vec![Guess {
                guess: possible_solutions.iter().next().unwrap(),
                variant: GuessType::Entropy {
                    entropy: 0.0,
                    solution_probability: 1.0,
                },
            }];
        }

        // let key = unordered_set_hash(possible_solutions);
        // if size >= 10
        //     && let Some(cached) = self.best_guess_cache.get(&key)
        // {
        //     return (*cached).clone();
        // }
        //

        let mut all_guesses: Vec<_> = self
            .wordlist()
            .par_iter()
            .map(|guess_str| {
                let mut pattern_counts: HashMap<[Feedback; 5], usize> = HashMap::new();

                let num_solutions = possible_solutions.len() as f64;
                let uniform_solution_proba = 1.0 / num_solutions;

                for solution in possible_solutions {
                    *pattern_counts
                        .entry(self.get_feedback(solution, guess_str)) // TODO: replace cached?
                        .or_insert(0) += 1;
                }
                let entropy = -pattern_counts
                    .iter()
                    .map(|(_, count)| {
                        let p = (*count as f64) / num_solutions;
                        p * p.log2()
                    })
                    .sum::<f64>();

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
            .collect();
        // biggest first
        all_guesses.sort_by(|a, b| b.quality().total_cmp(&a.quality()));

        // Take top k
        all_guesses.truncate(self.top_k);
        all_guesses
    }

    pub fn compute_best_initial_guess(&self) -> Guess {
        self.compute_best_guess(&HashSet::new(), self.wordlist(), true)
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
        self.used_guesses.insert(guess_result.guess);

        if self.verbose {
            // TODO:
            // println!("Result pattern: {:?}", guess_result.feedback);
            // println!("new possible_solutions: {:?}", self.possible_solutions);
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
        let guess = self.compute_best_guess(&self.used_guesses, self.possible_solutions(), true);
        if self.verbose {
            println!("\n > {guess:?}");
        }
        guess
    }
}

#[cfg(test)]
mod test {
    use serial_test::serial;

    mod guesser_logic {
        use super::super::MaxEntropyGuesser;
        use super::serial;
        use crate::{Feedback, GuessResult, Guesser};
        use hashbrown::HashSet;

        #[test]
        fn update_possible_solutions_filters_correctly() {
            let mut guesser = MaxEntropyGuesser::builder().wordlist_subset(100).build();

            let solution = "aargh";
            let guess = "abaca";

            let feedback = Feedback::compute(solution, guess);
            let result = GuessResult { guess, feedback };

            let before_count = guesser.possible_solutions().len();
            guesser.update_state(result);
            let after_count = guesser.possible_solutions().len();

            assert!(after_count < before_count);
            assert!(guesser.possible_solutions().contains(solution));
        }

        #[test]
        fn update_possible_solutions_eliminates_wrong() {
            let mut guesser = MaxEntropyGuesser::builder().wordlist_subset(100).build();

            let solution = "pulse";
            let guess = "tares";

            let feedback = Feedback::compute(solution, guess);
            let result = GuessResult { guess, feedback };

            guesser.update_state(result);

            for word in guesser.possible_solutions() {
                let computed = Feedback::compute(word, guess);
                assert_eq!(computed, feedback);
            }
        }

        #[test]
        #[serial]
        fn compute_best_guess_returns_valid_word() {
            let guesser = MaxEntropyGuesser::builder().wordlist_subset(100).build();

            let possible_solutions: HashSet<&str> =
                ["abcde", "fghij", "klmno"].iter().copied().collect();
            let best_guess = guesser.compute_best_guess(&possible_solutions);

            assert!(best_guess.get_string().len() == 5);
        }

        #[test]
        #[serial]
        fn compute_best_guess_single_solution() {
            let guesser = MaxEntropyGuesser::builder().wordlist_subset(100).build();

            let possible_solutions: HashSet<&str> = ["abcde"].iter().copied().collect();
            let best_guess = guesser.compute_best_guess(&possible_solutions);

            assert_eq!(best_guess.get_string(), "abcde");
        }

        #[test]
        #[serial]
        fn compute_best_guess_entropy_calculation() {
            let guesser = MaxEntropyGuesser::builder().wordlist_subset(100).build();

            let possible_solutions: HashSet<&str> = ["abcde", "fghij", "klmno", "pqrst"]
                .iter()
                .copied()
                .collect();
            let best_guess = guesser.compute_best_guess(&possible_solutions);

            let (entropy, _) = best_guess.unwrap_entropy();
            assert!(entropy >= 0.0);
            assert!(entropy <= (possible_solutions.len() as f64).log2());
        }

        #[test]
        #[serial]
        fn compute_best_guess_cache_hit() {
            let guesser = MaxEntropyGuesser::builder().wordlist_subset(50).build();

            let possible_solutions: HashSet<&str> = [
                "abcde", "fghij", "klmno", "pqrst", "abcde", "fghij", "klmno", "pqrst", "abcde",
                "fghij",
            ]
            .iter()
            .copied()
            .collect();

            let guess1 = guesser.compute_best_guess(&possible_solutions);

            let possible_solutions2: HashSet<&str> = [
                "abcde", "fghij", "klmno", "pqrst", "abcde", "fghij", "klmno", "pqrst", "abcde",
                "fghij",
            ]
            .iter()
            .copied()
            .collect();
            let guess2 = guesser.compute_best_guess(&possible_solutions2);

            assert_eq!(guess1.get_string(), guess2.get_string());
        }

        #[test]
        #[serial]
        fn compute_best_initial_guess_returns_valid_word() {
            let guesser = MaxEntropyGuesser::builder().wordlist_subset(100).build();

            let (score, guess) = guesser.compute_best_initial_guess(true);

            assert!(guess.get_string().len() == 5);
            assert!(score >= 0.0);
        }

        #[test]
        #[serial]
        fn compute_best_initial_guess_is_in_wordlist() {
            let guesser = MaxEntropyGuesser::builder().wordlist_subset(100).build();

            let (_, guess) = guesser.compute_best_initial_guess(true);

            assert!(guesser.wordlist().contains(guess.get_string()));
        }

        #[test]
        #[serial]
        fn compute_best_initial_guess_has_entropy() {
            let guesser = MaxEntropyGuesser::builder().wordlist_subset(100).build();

            let (_, guess) = guesser.compute_best_initial_guess(true);

            let (entropy, _) = guess.unwrap_entropy();
            assert!(entropy > 0.0);
        }
    }
    mod pattern_partitions {
        use super::super::MaxEntropyGuesser;
        use crate::Feedback;
        use hashbrown::HashSet;

        const TEST_WORDLIST: &[&str] = &["abcde", "fghij", "klmno", "pqrst", "abcde"];

        #[test]
        fn pattern_partitions_creates_buckets() {
            let guesser = MaxEntropyGuesser::new();
            let possible_solutions: HashSet<&str> = TEST_WORDLIST.iter().copied().collect();
            let guess = "xxxxx";

            let buckets = guesser.pattern_partitions(guess, &possible_solutions);

            assert_eq!(buckets.len(), 1);
            assert!(buckets.contains_key(&[Feedback::Wrong; 5]));
        }

        #[test]
        fn pattern_partitions_correct_bucket() {
            let guesser = MaxEntropyGuesser::new();
            let possible_solutions: HashSet<&str> =
                ["abcde", "abced", "edcba"].iter().copied().collect();
            let guess = "abcde";

            let buckets = guesser.pattern_partitions(guess, &possible_solutions);

            let correct_bucket = buckets.get(&[Feedback::Correct; 5]).unwrap();
            assert_eq!(correct_bucket.len(), 1);
            assert!(correct_bucket.contains(&"abcde"));
        }

        #[test]
        fn pattern_partitions_mixed_feedback() {
            let guesser = MaxEntropyGuesser::new();
            let possible_solutions: HashSet<&str> =
                ["abcde", "fghij", "xyzab"].iter().copied().collect();
            let guess = "cdefg";

            let buckets = guesser.pattern_partitions(guess, &possible_solutions);

            assert!(buckets.len() > 1);
            for (pattern, words) in &buckets {
                for word in words {
                    let computed = Feedback::compute(word, guess);
                    assert_eq!(*pattern, computed);
                }
            }
        }
    }
    mod cache_and_hash {
        use super::super::{MaxEntropyGuesser, unordered_set_hash};
        use super::serial;
        use crate::Guesser;
        use hashbrown::HashSet;

        // NOTE: hash function uses pointer addresses for &'static str! so we cannot hardcode words

        #[test]
        fn unordered_set_hash_same_set_same_hash() {
            let s1 = "abc";
            let s2 = "def";
            let s3 = "ghi";
            let set1: HashSet<&str> = [s1, s2, s3].iter().copied().collect();
            let set2: HashSet<&str> = [s1, s2, s3].iter().copied().collect();

            let hash1 = unordered_set_hash(&set1);
            let hash2 = unordered_set_hash(&set2);

            assert_eq!(hash1, hash2);
        }

        #[test]
        fn unordered_set_hash_different_sets_different_hash() {
            let s1 = "abc";
            let s2 = "def";
            let s3 = "ghi";
            let s4 = "jkl";
            let set1: HashSet<&str> = [s1, s2, s3].iter().copied().collect();
            let set2: HashSet<&str> = [s1, s2, s4].iter().copied().collect();

            let hash1 = unordered_set_hash(&set1);
            let hash2 = unordered_set_hash(&set2);

            assert_ne!(hash1, hash2);
        }

        #[test]
        fn unordered_set_hash_empty_set() {
            let set: HashSet<&str> = HashSet::new();
            let hash = unordered_set_hash(&set);
            assert_eq!(hash, 0);
        }

        #[test]
        fn unordered_set_hash_single_element() {
            let set: HashSet<&str> = ["abc"].iter().copied().collect();
            let hash = unordered_set_hash(&set);
            assert!(hash > 0);
        }

        #[test]
        fn cache_key_reorder() {
            let s1 = "apple";
            let s2 = "apply";
            let s3 = "apron";
            let s4 = "ample";

            let possible_solutions: HashSet<&str> = [s1, s2, s3, s4].iter().copied().collect();
            let key1 = unordered_set_hash(&possible_solutions);

            let possible_solutions2: HashSet<&str> = [s3, s2, s4, s1].iter().copied().collect();
            let key2 = unordered_set_hash(&possible_solutions2);

            assert_eq!(key1, key2);
        }

        #[test]
        fn cache_key_different() {
            let s1 = "apple";
            let s2 = "apply";
            let s3 = "apron";
            let s4 = "ample";

            let possible_solutions: HashSet<&str> = [s1, s2, s3, s4].iter().copied().collect();
            let key1 = unordered_set_hash(&possible_solutions);

            let possible_solutions2: HashSet<&str> = [s1, s2, s3].iter().copied().collect();
            let key2 = unordered_set_hash(&possible_solutions2);

            assert_ne!(key1, key2);
        }

        #[test]
        #[serial]
        fn cache_prevents_recomputation() {
            let guesser = MaxEntropyGuesser::builder().wordlist_subset(30).build();

            let possible_solutions: HashSet<&str> =
                guesser.wordlist().iter().take(20).copied().collect();

            let _guess1 = guesser.compute_best_guess(&possible_solutions);

            let cache_len_before = guesser.best_guess_cache.len();

            let possible_solutions2: HashSet<&str> =
                guesser.wordlist().iter().take(20).copied().collect();

            let _guess2 = guesser.compute_best_guess(&possible_solutions2);

            assert!(guesser.best_guess_cache.len() == cache_len_before);
        }

        #[test]
        #[serial]
        fn cache_not_used_for_small_sets() {
            let guesser = MaxEntropyGuesser::builder().wordlist_subset(100).build();

            let possible_solutions: HashSet<&'static str> =
                [*guesser.wordlist().iter().next().unwrap()]
                    .iter()
                    .copied()
                    .collect();
            let _guess1 = guesser.compute_best_guess(&possible_solutions);

            assert_eq!(guesser.best_guess_cache.len(), 0);
        }

        #[test]
        #[serial]
        fn cache_is_used_for_large_sets() {
            let guesser = MaxEntropyGuesser::builder().wordlist_subset(100).build();

            let possible_solutions: HashSet<&str> =
                guesser.wordlist().iter().take(30).copied().collect();
            let _guess1 = guesser.compute_best_guess(&possible_solutions);

            assert!(guesser.best_guess_cache.len() > 0);
        }
    }
}
