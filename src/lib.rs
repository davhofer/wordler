use hashbrown::HashSet;
use std::{collections::HashMap, u8, io::{self, Write, stdin, stdout}};
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

    pub fn play<G: Guesser>(&self, solution: &'static str, mut guesser: G) -> Option<u32> {
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

    pub fn interactive_solve<G: Guesser>(&self, mut guesser: G) {
        println!("Let's solve a wordle puzzle together.");
        println!("I will propose the words to guess, and you respond with the resulting color pattern. Please write the pattern as 5 single characters separated by spaces. Use C (correct) for green, M (misplaced) for yellow, and W (wrong) for grey.\nFor example, the pattern [Green, Grey, Yellow, Grey, Grey] should be denoted as 'C W M W W'.");

        let mut prior_guesses: Vec<GuessResult> = Vec::new(); 
        for i in 1..=6 {
            let guess = guesser.guess(&self.words, &prior_guesses);
            println!("Guess #{i}: {}", guess.guess);

            let pattern = get_userinput();
            let feedback_vec: Vec<_> = pattern.trim().split_whitespace().map(|c| {
                match c {
                    "C" => Feedback::Correct,
                    "M" => Feedback::Misplaced,
                    "W" => Feedback::Wrong,
                    _ => panic!("only enter 'C', 'M', or 'W'"),
                }
            }).collect();

            if feedback_vec.len() != 5 {
                panic!("you need to enter exactly 5 characters");
            }
            let mut feedback = [Feedback::Wrong; 5];
            for i in 0..5 {
                feedback[i] = feedback_vec[i];
            }
            prior_guesses.push(GuessResult { guess: guess.guess, feedback });
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
                print!("\rProgress: {:.2}%", ((i as f64)/total_games) * 100.0);
                io::stdout().flush().unwrap();
            }
        }

        let failed_games = failed.len();


        println!("\rProgress: 100%     ");
        io::stdout().flush().unwrap();
        println!("Benchmark complete!");
        println!("Games: {}, Avg guesses: {}, Failures: {} ({}%)", total_games, (total_guesses as f64)/total_games, failed_games, ((failed_games as f64)/total_games) * 100.0);


        println!("Num guesses histogram:");
        for i in 1..=6 {
            println!("{} guesses: {}", i, *guesses_hist.get(&i).unwrap_or(&0)) ;
        }
        println!("7+ guesses (fail): {}", failed_games);

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
    stdin().read_line(&mut s).expect("Did not enter a correct string");
    if let Some('\n')=s.chars().next_back() {
        s.pop();
    }
    if let Some('\r')=s.chars().next_back() {
        s.pop();
    }
    s
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
