use clap::{Parser, Subcommand, ValueEnum};
use std::{time::Instant};
use wordler::{
    FeedbackStorage, MaxEntropyGuesser, MinExpectedScoreGuesser, Wordle, find_word_in_wordlist,
};

const BENCH_SOLUTIONS: [&str; 3] = ["doved", "jings", "vaxes"];

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Command to execute
    #[command(subcommand)]
    command: Cmd,
}

/// Wordle solver
#[derive(Debug, Subcommand)]
#[command()]
enum Cmd {
    /// Interactively use the bot to solve a wordle game
    #[command()]
    Solve {
        /// Wordle guesser strategy
        #[arg(short, long, required = true)]
        guesser: GuesserTypeEnum,

        /// Initial guess
        #[arg(short, long)]
        initial_guess: Option<String>,
    },

    /// Benchmark the guesser strategies of the wordle bot
    #[command()]
    Benchmark {
        /// Wordle guesser strategy to benchmark. If not provided, all guessers will be
        /// benchmarked.
        #[arg(short, long)]
        guesser: Option<GuesserTypeEnum>,

        // TODO: keep this?
        /// Initial guess to use for benchmarking
        #[arg(short, long)]
        initial_guess: Option<String>,

        /// Max. number of games to run for benchmarking
        #[arg(short, long)]
        max_games: Option<usize>,
    },

    /// Precompute the best first guess
    #[command()]
    PrecomputeInitialGuess {},

    /// Precompute all pairwise feedback patterns
    #[command()]
    PrecomputePatterns {},

    #[command()]
    CustomBench {},

    /// Let the bot play a game
    #[command()]
    Play {
        /// Wordle guesser strategy
        #[arg(short, long, required = true)]
        guesser: GuesserTypeEnum,

        /// Initial guess
        #[arg(short, long)]
        initial_guess: Option<String>,

        /// Wordle solution. If not set, a random solution will be picked
        #[arg(short, long)]
        solution: Option<String>,
    },
}

#[derive(ValueEnum, Debug, Clone)]
#[value()]
enum GuesserTypeEnum {
    MaxEntropyGuesser,
    MinExpectedScoreGuesser,
}

// fn make_guesser(guesser_type: GuesserTypeEnum) -> Box<dyn Guesser> {
//     match guesser_type {
//         GuesserTypeEnum::MaxEntropyGuesser => Box::new(MaxEntropyGuesser::new()),
//         GuesserTypeEnum::MinExpectedScoreGuesser => Box::new(MinExpectedScoreGuesser::new()),
//     }
// }

fn play(
    guesser_arg: GuesserTypeEnum,
    initial_guess_arg: Option<String>,
    solution_arg: Option<String>,
) {
    let wordle = Wordle::new();

    let solution = if let Some(solution_str) = solution_arg {
        find_word_in_wordlist(solution_str, wordle.wordlist()).expect(
            "'{solution_str}' is not in the wordlist, choose a different word as the solution",
        )
    } else {
        let sol = wordle.random_word().expect("unable to choose random word");
        println!("Choosing random word as the solution: '{sol}'.");
        sol
    };

    // Set initial guess
    let initial_guess = initial_guess_arg.map(|item| {
        find_word_in_wordlist(item, wordle.wordlist()).expect(
            "'{solution_str}' is not in the wordlist, choose a different word as the solution",
        )
    });

    match guesser_arg {
        GuesserTypeEnum::MaxEntropyGuesser => {
            let guesser = MaxEntropyGuesser::builder()
                .verbose()
                .initial_guess_option(initial_guess)
                .build();

            if let Some(guesses) = wordle.play(solution, guesser) {
                println!("Solved wordle in {guesses} guesses!");
            } else {
                println!("Guesser did not find a solution!");
            };
        }
        GuesserTypeEnum::MinExpectedScoreGuesser => {
            let mut guesser = MinExpectedScoreGuesser::new();
            guesser.set_verbose();

            if let Some(guess) = initial_guess {
                guesser.set_initial_guess(guess);
            }
            if let Some(guesses) = wordle.play(solution, guesser) {
                println!("Solved wordle in {guesses} guesses!");
            } else {
                println!("Guesser did not find a solution!");
            };
        }
    };
}

fn main() {
    let args = Args::parse();

    match args.command {
        Cmd::Play {
            guesser,
            initial_guess,
            solution,
        } => {
            play(guesser, initial_guess, solution);
        }
        Cmd::Solve {
            guesser,
            initial_guess,
        } => {
            solve(guesser, initial_guess);
        }
        Cmd::Benchmark {
            guesser,
            initial_guess,
            max_games,
        } => {
            benchmark(guesser, max_games, initial_guess);
        }
        Cmd::PrecomputeInitialGuess {} => {
            compute_best_initial_guesses();
        }
        Cmd::CustomBench {} => {
            custom_bench();
        }
        Cmd::PrecomputePatterns {} => {
            precompute_patterns();
        }
    }
}

fn solve(guesser_arg: GuesserTypeEnum, initial_guess_arg: Option<String>) {
    let wordle = Wordle::new();

    // Set initial guess
    let initial_guess = initial_guess_arg.map(|item| {
        find_word_in_wordlist(item, wordle.wordlist()).expect(
            "'{solution_str}' is not in the wordlist, choose a different word as the solution",
        )
    });

    match guesser_arg {
        GuesserTypeEnum::MaxEntropyGuesser => {
            let guesser = MaxEntropyGuesser::builder()
                .verbose()
                .initial_guess_option(initial_guess)
                .build();
            wordle.interactive_solve(guesser);
        }
        GuesserTypeEnum::MinExpectedScoreGuesser => {
            let mut guesser = MinExpectedScoreGuesser::new();
            guesser.set_verbose();

            if let Some(guess) = initial_guess {
                guesser.set_initial_guess(guess);
            }
            wordle.interactive_solve(guesser);
        }
    };
}

fn benchmark(
    guesser_arg: Option<GuesserTypeEnum>,
    max_games: Option<usize>,
    initial_guess: Option<String>,
) {
    let wordle = Wordle::new();

    // Set initial guess
    let initial_guess = initial_guess.map(|item| {
        let err = format!(
            "'{item}' is not in the wordlist, choose a different word as the initial guess"
        );
        find_word_in_wordlist(item, wordle.wordlist()).expect(&err)
    });

    let make_entropy_guesser = || {
        MaxEntropyGuesser::builder()
            .initial_guess_option(initial_guess)
            .build()
    };
    let make_expected_score_guesser = || {
        let mut guesser = MinExpectedScoreGuesser::new();
        if let Some(guess) = initial_guess {
            guesser.set_initial_guess(guess);
        }
        guesser
    };

    match guesser_arg {
        Some(GuesserTypeEnum::MaxEntropyGuesser) => {
            println!("Benchmarking MaxEntropyGuesser");
            wordle.benchmark(make_entropy_guesser, max_games);
        }
        Some(GuesserTypeEnum::MinExpectedScoreGuesser) => {
            println!("Benchmarking MinExpectedScoreGuesser");
            wordle.benchmark(make_expected_score_guesser, max_games);
        }
        None => {
            println!("Benchmarking MaxEntropyGuesser");
            wordle.benchmark(make_entropy_guesser, max_games);
            println!("Benchmarking MinExpectedScoreGuesser");
            wordle.benchmark(make_expected_score_guesser, max_games);
        }
    };
}

fn custom_bench() {
    let wordle = Wordle::new();

    let feedback_storage = FeedbackStorage::load().unwrap();
    let guesser = MaxEntropyGuesser::builder().precomputed_patterns(feedback_storage).build();

    let start = Instant::now();

    let _ = wordle.play("oomph", guesser);

    let duration = (Instant::now() - start).as_secs_f32();
    println!("took {duration:.2}s")

    //
    //
    // println!("Custom benchmark for MaxEntropyGuesser");
    //
    // println!("Playing {} games", BENCH_SOLUTIONS.len());
    // for solution in BENCH_SOLUTIONS {
    //     let mut guesser = MaxEntropyGuesser::new();
    //     if use_feedback_storage {
    //         let feedback_storage = FeedbackStorage::load().unwrap();
    //         guesser.use_precomputed_patterns(feedback_storage);
    //     }
    //     let _ = wordle.play(&solution, guesser);
    // }
    //
    // let wordlist_size: usize = 800;
    //
    // println!("Computing best l2 guess with wordlist_size={wordlist_size}");
    //
    // let mut guesser = MaxEntropyGuesser::new();
    // guesser.use_wordlist_subset(wordlist_size);
    //
    //
    // if use_feedback_storage {
    //     let feedback_storage = FeedbackStorage::load().unwrap();
    //     guesser.use_precomputed_patterns(feedback_storage);
    // }
    //
    // let _ = guesser.compute_best_initial_guess();
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

    // guesser.use_wordlist_subset(1500);

    // 1000^3 <-> 15s
    // 1000 <-> 2.46
    //
    // 14000 <-> 36s
    // 14000^3 <-> 49121s = 818m = 13.6h
    //
    // 2000^3 <-> 120s = 2m (actually was 150s)
    //
    // 3000^3 <-> 404s = 7m (actually was 521s)

    let start = Instant::now();
    let (combined_entropy, best_l2_guess) = guesser.compute_best_initial_guess();
    let duration = start.elapsed().as_secs_f64();
    println!("{best_l2_guess:?}\nExpected information gain after 2 rounds: {combined_entropy}");
    println!("took {duration:.2}s");
}

fn precompute_patterns() {
    let start = Instant::now();
    if let Err(e) = FeedbackStorage::build_and_save() {
        println!("Error while saving feedback pattern storage: {e}")
    } else {
        println!("Feedback pattern storage saved successfully!");
    }
    let duration = (Instant::now() - start).as_secs();
    println!("took {duration:.2}s")
}
