use hashbrown::HashSet;
use rand::{rng, seq::IteratorRandom};
use std::fs::File;
use std::path::PathBuf;
use std::{
    collections::HashMap,
    io::{self, Read, Write, stdin, stdout},
};
use wincode::{self, SchemaRead, SchemaWrite, containers, len::BincodeLen};

mod guessers;
pub use guessers::{MaxEntropyGuesser, MinExpectedScoreGuesser};

/// Wordlist containing all possible guesses and solutions.
const WORDS: &str = include_str!("../data/words.txt");

pub fn find_string_in_wordlist(
    word: String,
    wordlist: &HashSet<&'static str>,
) -> Option<&'static str> {
    wordlist
        .iter()
        .filter(|&&candidate| candidate == word)
        .next()
        .map(|item| *item)
}

#[derive(Debug)]
pub struct Wordle {
    words: HashSet<&'static str>,
}

impl Wordle {
    pub fn new() -> Self {
        let words: HashSet<&str> = WORDS.split_whitespace().collect();
        Self { words }
    }

    pub fn play<G: Guesser>(&self, solution: &'static str, mut guesser: G) -> Option<u32> {
        for i in 1..=10 {
            let guess = guesser.guess().get_string();
            if guess == solution {
                return Some(i);
            }
            let feedback = Feedback::compute(solution, guess);
            guesser.update_state(GuessResult { guess, feedback });
        }
        None
    }

    pub fn interactive_solve<G: Guesser>(&self, mut guesser: G) {
        println!("Let's solve a wordle puzzle together.");
        println!(
            "I will propose the words to guess, and you respond with the resulting color pattern. Please write the pattern as 5 single characters separated by spaces. Use C (correct) for green, M (misplaced) for yellow, and W (wrong) for grey.\nFor example, the pattern [Green, Grey, Yellow, Grey, Grey] should be denoted as 'C W M W W'."
        );

        for i in 1..=16 {
            let guess = guesser.guess().get_string();
            println!("\nGuess #{i}: {}", guess);

            let pattern = get_userinput();
            let feedback_vec: Vec<_> = pattern
                .trim()
                .split_whitespace()
                .map(|c| match c {
                    "C" => Feedback::Correct,
                    "M" => Feedback::Misplaced,
                    "W" => Feedback::Wrong,
                    _ => panic!("only enter 'C', 'M', or 'W'"),
                })
                .collect();

            if feedback_vec.len() != 5 {
                panic!("you need to enter exactly 5 characters");
            }
            let mut feedback = [Feedback::Wrong; 5];
            for i in 0..5 {
                feedback[i] = feedback_vec[i];
            }
            guesser.update_state(GuessResult { guess, feedback });

            if guesser.possible_solutions().len() == 1 {
                let ans = *guesser.possible_solutions().iter().next().unwrap();
                println!("\nSolution must be: {ans}");
                return;
            }
        }
    }

    pub fn random_word(&self) -> Option<&'static str> {
        self.words.iter().choose(&mut rng()).map(|word| *word)
    }

    pub fn wordlist(&self) -> &HashSet<&'static str> {
        &self.words
    }

    pub fn benchmark<F, G>(&self, make_guesser: F, max_games: Option<usize>)
    where
        F: Fn() -> G,
        G: Guesser,
    {
        // TODO: compute average information gain after n rounds (to check if it matches with
        // expectation for initial guess)

        let max_games = usize::min(max_games.unwrap_or(self.words.len()), self.words.len());
        let total_games = max_games as f64;

        let mut guesses_hist: HashMap<u32, u32> = HashMap::new();

        let mut failed = Vec::new();

        let mut total_guesses = 0;
        for (i, &solution) in self.words.iter().enumerate() {
            if i >= max_games {
                break;
            }

            let guesser = make_guesser();
            let guesses = self.play(solution, guesser).unwrap_or(0);

            if guesses == 0 {
                failed.push(solution);
            } else {
                *guesses_hist.entry(guesses).or_insert(0) += 1;
                total_guesses += guesses;
            }

            if i % 10 == 0 {
                print!("\rProgress: {:.2}%", ((i as f64) / total_games) * 100.0);
                io::stdout().flush().unwrap();
            }
        }

        let failed_games = failed.len();

        println!("\rProgress: 100%     ");
        io::stdout().flush().unwrap();
        println!("Benchmark complete!");
        println!(
            "Games: {}, Avg guesses: {}, Failures: {} ({}%)",
            total_games,
            (total_guesses as f64) / total_games,
            failed_games,
            ((failed_games as f64) / total_games) * 100.0
        );

        println!("Num guesses histogram:");
        for i in 1..=10 {
            println!("{} guesses: {}", i, *guesses_hist.get(&i).unwrap_or(&0));
        }
        println!("11+ guesses (fail): {}", failed_games);

        println!("\nFailed words:");
        for w in failed {
            println!("{w}");
        }
    }
}

fn get_userinput() -> String {
    let mut s = String::new();
    print!("Please enter the pattern using C, M, and W: ");
    let _ = stdout().flush();
    stdin()
        .read_line(&mut s)
        .expect("Did not enter a correct string");
    if let Some('\n') = s.chars().next_back() {
        s.pop();
    }
    if let Some('\r') = s.chars().next_back() {
        s.pop();
    }
    s
}

#[derive(SchemaRead, SchemaWrite, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Feedback {
    Correct,   // Green
    Misplaced, // Yellow
    Wrong,     // Gray
}

#[derive(SchemaRead, SchemaWrite)]
struct FeedbackVec {
    #[wincode(with = "containers::Vec<[Feedback; 5], BincodeLen<{ 2 * 1024 * 1024 * 1024 }>>")]
    pub entries: Vec<[Feedback; 5]>,
}

impl FeedbackVec {
    pub fn load(path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        // 1. Read file contents into a buffer
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        // required size = 2 * 1024 * 1024 * 1024usize; // 2 GiB

        // 2. Deserialize into the storage type
        let vec: Self = wincode::deserialize(&buffer)?;

        Ok(vec)
    }

    pub fn store(&self, path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = wincode::serialize(self)?;

        let mut file = File::create(path)?;
        file.write_all(&encoded)?;

        Ok(())
    }
}

pub struct FeedbackStorage {
    wordlist_size: usize,
    data: FeedbackVec,
    word_to_idx: HashMap<&'static str, usize>,
}

impl FeedbackStorage {
    /// Hardcoded path where FeedbackStorage is saved
    fn get_storage_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("feedback_patterns.storage")
    }

    /// Retrieve precomputed feedback pattern
    pub fn get(&self, solution: &'static str, guess: &'static str) -> Option<[Feedback; 5]> {
        let sol_idx = self.word_to_idx.get(solution)?;
        let guess_idx = self.word_to_idx.get(guess)?;
        let idx = guess_idx * self.wordlist_size + sol_idx;
        assert!(
            idx < self.data.entries.len(),
            "precomputed FeedbackVec is to small, has it been properly initialized?"
        );
        Some(self.data.entries[idx])
    }

    /// Compute and save feedback pattern for all (solution, guess) pairs
    pub fn build_and_save() -> Result<(), Box<dyn std::error::Error>> {
        let words: Vec<&'static str> = WORDS.split_whitespace().collect();
        let n = words.len();
        let mut entries: Vec<[Feedback; 5]> = std::vec::Vec::with_capacity(n * n);

        for &guess in &words {
            for &solution in &words {
                entries.push(Feedback::compute(solution, guess));
            }
        }

        FeedbackVec { entries }.store(Self::get_storage_path())
    }

    /// Load the precomputed feedback storage and initialize it
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_from_path(Self::get_storage_path())
    }

    fn load_from_path(path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        println!("Loading FeedbackStorage...");
        let words: Vec<&'static str> = WORDS.split_whitespace().collect();
        let word_to_idx: HashMap<&str, usize> =
            words.iter().enumerate().map(|(i, &w)| (w, i)).collect();
        let feedback_vec = FeedbackVec::load(path)?;

        Ok(Self {
            wordlist_size: words.len(),
            data: feedback_vec,
            word_to_idx,
        })
    }
}

impl Feedback {
    pub fn compute(solution: &str, guess: &str) -> [Self; 5] {
        let mut used = [false; 5];
        let mut mask = [Self::Wrong; 5];

        let guess_vec: Vec<_> = guess.chars().collect();

        for (i, letter) in solution.char_indices() {
            if letter == guess_vec[i] {
                used[i] = true;
                mask[i] = Self::Correct;
            }
        }

        for (i, letter) in guess_vec.iter().enumerate() {
            if mask[i] == Self::Correct {
                continue;
            }

            for (j, c) in solution.char_indices() {
                if !used[j] && *letter == c {
                    used[j] = true;
                    mask[i] = Self::Misplaced;
                    break;
                }
            }
        }
        mask
    }
}

#[derive(Debug, Clone)]
pub enum GuessType {
    Entropy {
        entropy: f64,
        solution_probability: f64,
    },
    ExpectedScore {
        score: f64,
    },
}

#[derive(Debug, Clone)]
pub struct Guess {
    pub guess: &'static str,
    pub variant: GuessType,
}

impl Guess {
    // TODO: different ways of computing guess quality from entropy & solution_probability?
    pub fn quality(&self) -> f64 {
        match self.variant {
            GuessType::Entropy {
                entropy,
                solution_probability,
            } => entropy + solution_probability,
            GuessType::ExpectedScore { score } => score, // TODO: bad interface, for this variant
                                                         // lower quality is better...
        }
    }

    pub fn get_variant(&self) -> GuessType {
        self.variant.clone()
    }

    pub fn get_string(&self) -> &'static str {
        self.guess
    }

    pub fn unwrap_entropy(&self) -> (f64, f64) {
        match self.variant {
            GuessType::Entropy {
                entropy,
                solution_probability,
            } => (entropy, solution_probability),
            _ => panic!("Called unwrap_entropy on a non-Entropy guess!"),
        }
    }

    pub fn unwrap_expected_score(&self) -> f64 {
        match self.variant {
            GuessType::ExpectedScore { score } => score,
            _ => panic!("Called unwrap_expected_score on a non-ExpectedScore guess!"),
        }
    }
}

pub struct GuessResult {
    guess: &'static str,
    feedback: [Feedback; 5],
}

pub trait Guesser {
    /// Produce a new guess based on the wordlist and set of currently possible solutions
    fn guess(&self) -> Guess;

    fn guesses_made(&self) -> u32;

    /// Update internal state of guesser (e.g. possible_solutions, guesses_made) using the most
    /// recent guess
    fn update_state(&mut self, guess_result: GuessResult);

    // Set of allowed guesses
    fn wordlist(&self) -> &HashSet<&'static str>;

    // Set of currently possible solutions
    fn possible_solutions(&self) -> &HashSet<&'static str>;
}

// ADAPTED FROM https://github.com/jonhoo/roget/blob/main/src/lib.rs
#[cfg(test)]
macro_rules! mask {
    (C) => {$crate::Feedback::Correct};
    (M) => {$crate::Feedback::Misplaced};
    (W) => {$crate::Feedback::Wrong};
    ($($c:tt)+) => {[
        $(mask!($c)),+
    ]}
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    mod play {
        use super::serial;
        use crate::{MaxEntropyGuesser, Wordle};

        #[test]
        #[serial]
        fn guess_found1() {
            let wordle = Wordle::new();
            let guesser = MaxEntropyGuesser::builder().initial_guess("tares").build();
            let sol_found = if let Some(_) = wordle.play("tares", guesser) {
                true
            } else {
                false
            };
            assert!(sol_found);
        }

        #[test]
        #[serial]
        fn guess_found2() {
            let wordle = Wordle::new();
            let guesser = MaxEntropyGuesser::builder().initial_guess("tares").build();
            let sol_found = if let Some(_) = wordle.play("oomph", guesser) {
                true
            } else {
                false
            };
            assert!(sol_found);
        }

        #[test]
        #[serial]
        fn guess_found3() {
            let wordle = Wordle::new();
            let guesser = MaxEntropyGuesser::builder().initial_guess("tares").build();
            let sol_found = if let Some(_) = wordle.play("aargh", guesser) {
                true
            } else {
                false
            };
            assert!(sol_found);
        }

        #[test]
        #[serial]
        fn guess_found4() {
            let wordle = Wordle::new();
            let guesser = MaxEntropyGuesser::builder().initial_guess("tares").build();
            let sol_found = if let Some(_) = wordle.play("bobby", guesser) {
                true
            } else {
                false
            };
            assert!(sol_found);
        }

        #[test]
        #[serial]
        fn guess_found5() {
            let wordle = Wordle::new();
            let guesser = MaxEntropyGuesser::builder().initial_guess("tares").build();
            let sol_found = if let Some(_) = wordle.play("pulse", guesser) {
                true
            } else {
                false
            };
            assert!(sol_found);
        }
    }
    mod compute {
        use crate::Feedback;

        #[test]
        fn all_green() {
            assert_eq!(Feedback::compute("abcde", "abcde"), mask![C C C C C]);
        }

        #[test]
        fn all_gray() {
            assert_eq!(Feedback::compute("abcde", "fghij"), mask![W W W W W]);
        }

        #[test]
        fn all_yellow() {
            assert_eq!(Feedback::compute("abcde", "eabcd"), mask![M M M M M]);
        }

        #[test]
        fn repeat_green() {
            assert_eq!(Feedback::compute("aabbb", "aaccc"), mask![C C W W W]);
        }

        #[test]
        fn repeat_yellow() {
            assert_eq!(Feedback::compute("aabbb", "ccaac"), mask![W W M M W]);
        }

        #[test]
        fn repeat_some_green() {
            assert_eq!(Feedback::compute("aabbb", "caacc"), mask![W C M W W]);
        }

        #[test]
        fn dremann_from_chat() {
            assert_eq!(Feedback::compute("azzaz", "aaabb"), mask![C M W W W]);
        }

        #[test]
        fn itsapoque_from_chat() {
            assert_eq!(Feedback::compute("baccc", "aaddd"), mask![W C W W W]);
        }

        #[test]
        fn ricoello_from_chat() {
            assert_eq!(Feedback::compute("abcde", "aacde"), mask![C W C C C]);
        }
    }
    mod feedback_vec {
        use crate::{Feedback, FeedbackVec};
        use tempfile::NamedTempFile;

        #[test]
        fn serialize_deserialize_1k() {
            let n = 1000;
            let feedback_vec = FeedbackVec {
                entries: vec![[Feedback::Correct; 5]; n * n],
            };

            let temp_file = NamedTempFile::new().unwrap();
            let path = temp_file.into_temp_path();

            let res = feedback_vec.store(path.to_path_buf());
            if let Ok(_) = res {
            } else {
                assert!(false);
            }
            drop(feedback_vec);

            if let Ok(feedback_vec) = FeedbackVec::load(path.to_path_buf()) {
                assert_eq!(feedback_vec.entries[0], [Feedback::Correct; 5]);
                assert_eq!(
                    feedback_vec.entries[(n / 2) * (n / 2)],
                    [Feedback::Correct; 5]
                );
                assert_eq!(feedback_vec.entries[n * n - 1], [Feedback::Correct; 5]);
            } else {
                assert!(false);
            }
        }

        #[test]
        fn serialize_deserialize_5k() {
            let n = 5000;
            let feedback_vec = FeedbackVec {
                entries: vec![[Feedback::Correct; 5]; n * n],
            };
            let temp_file = NamedTempFile::new().unwrap();
            let path = temp_file.into_temp_path();

            let res = feedback_vec.store(path.to_path_buf());
            if let Ok(_) = res {
            } else {
                assert!(false);
            }
            drop(feedback_vec);

            if let Ok(feedback_vec) = FeedbackVec::load(path.to_path_buf()) {
                assert_eq!(feedback_vec.entries[0], [Feedback::Correct; 5]);
                assert_eq!(
                    feedback_vec.entries[(n / 2) * (n / 2)],
                    [Feedback::Correct; 5]
                );
                assert_eq!(feedback_vec.entries[n * n - 1], [Feedback::Correct; 5]);
            } else {
                assert!(false);
            }
        }
    }
    mod storage {
        use crate::{Feedback, FeedbackStorage, FeedbackVec};
        use std::fs::File;
        use std::io::Write;
        use tempfile::TempDir;

        #[test]
        fn load_nonexistent_file_returns_error() {
            let temp_dir = TempDir::new().unwrap();
            let non_existent_path = temp_dir.path().join("nonexistent.storage");

            let result = FeedbackStorage::load_from_path(non_existent_path);

            assert!(result.is_err());
        }

        #[test]
        fn load_corrupted_file_returns_error() {
            let temp_dir = TempDir::new().unwrap();
            let corrupted_path = temp_dir.path().join("corrupted.storage");

            let mut file = File::create(&corrupted_path).unwrap();
            file.write_all(b"this is not valid wincode data").unwrap();

            let result = FeedbackStorage::load_from_path(corrupted_path);

            assert!(result.is_err());
        }

        #[test]
        fn load_partial_data_returns_error() {
            let temp_dir = TempDir::new().unwrap();
            let partial_path = temp_dir.path().join("partial.storage");

            let mut file = File::create(&partial_path).unwrap();
            file.write_all(b"partial").unwrap();

            let result = FeedbackStorage::load_from_path(partial_path);

            assert!(result.is_err());
        }

        #[test]
        fn feedback_vec_entries_accessible() {
            let n = 100;
            let entries = vec![[Feedback::Wrong; 5]; n * n];
            let feedback_vec = FeedbackVec { entries };

            assert_eq!(feedback_vec.entries.len(), n * n);
            assert_eq!(feedback_vec.entries[0], [Feedback::Wrong; 5]);
            assert_eq!(feedback_vec.entries[n * n - 1], [Feedback::Wrong; 5]);
        }

        #[test]
        fn feedback_vec_serialize_deserialize_roundtrip() {
            let n = 50;
            let entries: Vec<[Feedback; 5]> = (0..n * n)
                .map(|i| {
                    if i % 3 == 0 {
                        [Feedback::Correct; 5]
                    } else if i % 3 == 1 {
                        [Feedback::Misplaced; 5]
                    } else {
                        [Feedback::Wrong; 5]
                    }
                })
                .collect();
            let feedback_vec = FeedbackVec { entries };

            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("roundtrip.storage");
            feedback_vec.store(path.clone()).unwrap();

            let loaded = FeedbackVec::load(path).unwrap();

            assert_eq!(loaded.entries.len(), n * n);
            for (i, entry) in loaded.entries.iter().enumerate().take(100) {
                let original = if i % 3 == 0 {
                    [Feedback::Correct; 5]
                } else if i % 3 == 1 {
                    [Feedback::Misplaced; 5]
                } else {
                    [Feedback::Wrong; 5]
                };
                assert_eq!(*entry, original);
            }
        }
    }
}
