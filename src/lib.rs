use hashbrown::HashSet;
use std::u16;
use rand::{rng, seq::IteratorRandom};

mod guessers;
pub use guessers::MaxEntropyGuesser;

/// Wordlist containing all possible guesses and solutions.
const WORDS: &str = include_str!("../words.txt");

#[derive(Debug)]
pub struct Wordle {
    words: HashSet<&'static str>,
}

impl Wordle {
    pub fn new() -> Self {
        let words: HashSet<&str> = WORDS.split_whitespace().collect();
        Self { words }
    }

    pub fn play<G: Guesser>(&self, solution: &'static str, mut guesser: G) -> Option<u16> {
        let mut prior_guesses: Vec<GuessResult> = Vec::new(); 
        for i in 1..=6 {
            let guess = guesser.guess(&self.words, &prior_guesses);
            if guess.guess == solution {
                return Some(i);
            }
            let feedback = compute_feedback(solution, &guess.guess);
            prior_guesses.push(GuessResult { guess: guess.guess, feedback });
        }
        None 
    }

    pub fn random_word(&self) -> Option<&'static str> {
        self.words.iter().choose(&mut rng()).map(|word| *word)
    }

}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Feedback {
    Correct, // Green
    Misplaced, // Yellow
    Wrong, // Gray
}

pub fn compute_feedback(solution: &str, guess: &str) -> [Feedback; 5] {
    let mut used = [false; 5];
    let mut mask = [Feedback::Wrong; 5];

    let guess_vec: Vec<_> = guess.chars().collect();

    for (i, letter) in solution.char_indices() {
        if letter == guess_vec[i] {
            used[i] = true;
            mask[i] = Feedback::Correct;
        }
    }

    for (i, letter) in guess_vec.iter().enumerate() {
        if mask[i] == Feedback::Correct {
            continue;
        }

        for (j, c) in solution.char_indices() {
            if !used[j] && *letter == c {
                used[j] = true;
                mask[i] = Feedback::Misplaced;
                break;
            }
        }
    }
    mask
}

// TODO: make guess a trait and implement different ways of computing guess quality? or do this in
// a different place?
#[derive(Debug)]
pub struct Guess {
    guess: String,
    entropy: f64,
    solution_probability: f64,
}

impl PartialEq for Guess {
    fn eq(&self, other: &Self) -> bool {
        self.guess == other.guess
    }
    fn ne(&self, other: &Self) -> bool {
        self.guess != other.guess
    }
}


impl Guess {
    pub fn quality(&self) -> f64 {
        // TODO: other ways of computing guess quality?
        self.entropy + self.solution_probability
    }
}

pub struct GuessResult {
    guess: String,
    feedback: [Feedback; 5],
}

pub trait Guesser {
    
    /// Produce a new guess given the wordlist and prior guesses.
    ///
    /// (Note: the prior guesses are used to compute the list of possible solutions inside of
    /// `guess()`.)
    fn guess(&mut self, wordlist: &HashSet<&str>, prior_guesses: &Vec<GuessResult>) -> Guess;
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
    mod compute {
        use crate::compute_feedback;

        #[test]
        fn all_green() {
            assert_eq!(compute_feedback("abcde", "abcde"), mask![C C C C C]);
        }

        #[test]
        fn all_gray() {
            assert_eq!(compute_feedback("abcde", "fghij"), mask![W W W W W]);
        }

        #[test]
        fn all_yellow() {
            assert_eq!(compute_feedback("abcde", "eabcd"), mask![M M M M M]);
        }

        #[test]
        fn repeat_green() {
            assert_eq!(compute_feedback("aabbb", "aaccc"), mask![C C W W W]);
        }

        #[test]
        fn repeat_yellow() {
            assert_eq!(compute_feedback("aabbb", "ccaac"), mask![W W M M W]);
        }

        #[test]
        fn repeat_some_green() {
            assert_eq!(compute_feedback("aabbb", "caacc"), mask![W C M W W]);
        }

        #[test]
        fn dremann_from_chat() {
            assert_eq!(compute_feedback("azzaz", "aaabb"), mask![C M W W W]);
        }

        #[test]
        fn itsapoque_from_chat() {
            assert_eq!(compute_feedback("baccc", "aaddd"), mask![W C W W W]);
        }

        #[test]
        fn ricoello_from_chat() {
            assert_eq!(compute_feedback("abcde", "aacde"), mask![C W C C C]);
        }
    }
}
