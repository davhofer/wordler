use wordler::{Wordle, MaxEntropyGuesser};
use std::{env, time::{Duration, Instant}};

fn main() {
    // TODO: make a clean cli
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        match &args[1][..] {
            // TODO: allow setting the initial guess
            "play-random" => {
                let initial_guess = Some("tares");
                play_random_wordle(initial_guess);
            },
            "benchmark" => {
                // TODO: allow setting max number of games
                // and initial guess
                let initial_guess = Some("tares");
                let max_games = None;
                benchmark(max_games, initial_guess);
            },
            "precompute-guesses" => compute_best_initial_guesses(),
            "solve" => solve(),
            _ => println!("Please provide one of the following commands: 'play-random', 'benchmark', 'solve', or 'precompute-guesses'"),
        }
    } else {
        println!("Please provide a command: 'play-random', 'benchmark', 'solve', or 'precompute-guesses'");
    }
}

fn solve() {
    let wordle = Wordle::new();
    let guesser = MaxEntropyGuesser::new();
    wordle.interactive_solve(guesser);
}

fn benchmark(max_games: Option<usize>, initial_guess: Option<&'static str>) {
    let wordle = Wordle::new();


    wordle.benchmark(|| {
        let mut guesser = MaxEntropyGuesser::new();
        if let Some(guess) = initial_guess {
            guesser.set_initial_guess(guess);
        }
        guesser
    }, max_games);
}

fn play_random_wordle(initial_guess: Option<&'static str>) {
    let wordle = Wordle::new();

    let solution = wordle.random_word().expect("unable to choose random word");
    println!("Choosing random word as solution: {solution}\n");

    let mut guesser = MaxEntropyGuesser::new();

    // Set initial guess
    if let Some(guess) = initial_guess {
        guesser.set_initial_guess(guess);
    }

    if let Some(guesses) = wordle.play(solution, guesser) {
        println!("Solved wordle in {guesses} guesses!");
    } else {
        println!("Guesser did not find a solution!");
    }
}

fn compute_best_initial_guesses() {
    let wordle = Wordle::new();

    println!("Computing best initial guesses for MaxEntropyGuesser.");

    println!("\nBest guess considering expected information gain after 1 round:");
    let guesser = MaxEntropyGuesser::new();
    let start = Instant::now();
    let best_l1_guess = guesser.compute_best_guess(wordle.wordlist());
    let duration = start.elapsed().as_secs_f64();
    println!("{best_l1_guess:?}");
    println!("took {duration:.2}s");


    println!("\nBest guess considering expected information gain after 2 rounds:");
    let guesser = MaxEntropyGuesser::new();
    let start = Instant::now();
    let (combined_entropy, best_l2_guess) = guesser.compute_best_initial_guess(wordle.wordlist());
    let duration = start.elapsed().as_secs_f64();
    println!("{best_l2_guess:?}\nExpected information gain after 2 rounds: {combined_entropy}");
    println!("took {duration:.2}s");

}
