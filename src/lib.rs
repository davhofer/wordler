use hashbrown::HashSet;
use rand::{rng, seq::IteratorRandom};
use std::fs::File;
use std::path::PathBuf;
use std::{
    collections::HashMap,
    io::{self, Read, Write, stdin, stdout},
};
use wincode::{self, containers, len::BincodeLen, SchemaRead, SchemaWrite};
// use wincode::containers::Vec;

mod guessers;
pub use guessers::{MaxEntropyGuesser, MinExpectedScoreGuesser};

/// Wordlist containing all possible guesses and solutions.
const WORDS: &str = include_str!("../data/words.txt");

pub fn find_word_in_wordlist(
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

        for i in 1..=6 {
            let guess = guesser.guess().get_string();
            println!("Guess #{i}: {}", guess);

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

    fn get_storage_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("feedback_patterns.storage")
    }

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

    /// Load the precomputed feedback storage from a file and initialize it
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let words: Vec<&'static str> = WORDS.split_whitespace().collect();
        let word_to_idx: HashMap<&str, usize> =
            words.iter().enumerate().map(|(i, &w)| (w, i)).collect();
        // TODO: load from file
        let feedback_vec = FeedbackVec::load(Self::get_storage_path())?;

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
enum GuessType {
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
    guess: &'static str,
    variant: GuessType,
}

impl Guess {
    // TODO: different ways of computing guess quality?
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
    /// Produce a new guess given the wordlist and prior guesses.
    ///
    /// (Note: the prior guesses are used to compute the list of possible solutions inside of
    /// `guess()`.)

    fn guess(&self) -> Guess;

    fn guesses_made(&self) -> u32;

    fn update_state(&mut self, guess_result: GuessResult);

    // set of allowed guesses
    fn wordlist(&self) -> &HashSet<&'static str>;

    // set of currently possible solutions
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
    mod play {
        use crate::{Wordle, MaxEntropyGuesser};

        #[test]
        fn guess_found1() {
            let wordle = Wordle::new();
            let guesser = MaxEntropyGuesser::builder()
                .initial_guess("tares")
                .build();
            let sol_found = if let Some(_) = wordle.play("tares", guesser) {
                true
            } else {
                false
            };
            assert!(sol_found);
        }

        #[test]
        fn guess_found2() {
            let wordle = Wordle::new();
            let guesser = MaxEntropyGuesser::builder()
                .initial_guess("tares")
                .build();
            let sol_found = if let Some(_) = wordle.play("oomph", guesser) {
                true
            } else {
                false
            };
            assert!(sol_found);
        }

        #[test]
        fn guess_found3() {
            let wordle = Wordle::new();
            let guesser = MaxEntropyGuesser::builder()
                .initial_guess("tares")
                .build();
            let sol_found = if let Some(_) = wordle.play("aargh", guesser) {
                true
            } else {
                false
            };
            assert!(sol_found);
        }

        #[test]
        fn guess_found4() {
            let wordle = Wordle::new();
            let guesser = MaxEntropyGuesser::builder()
                .initial_guess("tares")
                .build();
            let sol_found = if let Some(_) = wordle.play("bobby", guesser) {
                true
            } else {
                false
            };
            assert!(sol_found);
        }

        #[test]
        fn guess_found5() {
            let wordle = Wordle::new();
            let guesser = MaxEntropyGuesser::builder()
                .initial_guess("tares")
                .build();
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
        use std::path::PathBuf;
        use crate::{FeedbackVec, Feedback};

        #[test]
        fn serialize_deserialize() {
            let feedback_vec = FeedbackVec { entries: vec![[Feedback::Correct; 5]; 1024 * 1024 * 1024] };
            let path = PathBuf::from("tmp_feedback.storage");
            let res = feedback_vec.store(path);
            if let Ok(_) = res {} else {
                assert!(false);
            }
            drop(feedback_vec);

            let path = PathBuf::from("tmp_feedback.storage");
            if let Ok(feedback_vec) = FeedbackVec::load(path) {
                assert_eq!(feedback_vec.entries[0], [Feedback::Correct; 5]);
                assert_eq!(feedback_vec.entries[1024 * 1024], [Feedback::Correct; 5]);
                assert_eq!(feedback_vec.entries[1024 * 1024 * 1024 - 1], [Feedback::Correct; 5]);
            } else {
                assert!(false);
            }
        }
    }
}
