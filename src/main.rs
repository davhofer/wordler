use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;
use wordler::{
    FeedbackStorage, Guess, MaxEntropyGuesser, MinExpectedScoreGuesser, Wordle,
    find_string_in_wordlist,
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Command to execute
    #[command(subcommand)]
    command: Cmd,
}

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

        /// Only relevant for MinExpectedScoreGuesser. 
        /// Only the top_k guesses by entropy will be considered in each step.
        #[arg(short, long, default_value_t = 20)]
        top_k: usize,

        /// Load precomputed Feedback patterns for all (solution, guess) pairs
        #[arg(short, long, default_value_t = false)]
        load_feedback_patterns: bool,
    },

    /// Benchmark different guesser strategies
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

        /// Only relevant for MinExpectedScoreGuesser. 
        /// Only the top_k guesses by entropy will be considered in each step.
        #[arg(short, long, default_value_t = 20)]
        top_k: usize,
    },

    /// Perform expensive precomputations and store the results for later use
    #[command(subcommand)]
    Precompute(PrecomputeCmd),

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

        /// Only relevant for MinExpectedScoreGuesser. 
        /// Only the top_k guesses by entropy will be considered in each step.
        #[arg(short, long, default_value_t = 20)]
        top_k: usize,

        /// Load precomputed Feedback patterns for all (solution, guess) pairs
        #[arg(short, long, default_value_t = false)]
        load_feedback_patterns: bool,
    },
}

#[derive(Debug, Subcommand)]
#[command()]
enum PrecomputeCmd {
    /// Precompute the feedback patterns for all (solution, guess) pairs
    #[command()]
    FeedbackPatterns {},

    /// Precompute the best initial guess for a specific solver
    #[command()]
    InitialGuess {
        #[arg(short, long, required = true)]
        guesser: GuesserTypeEnum,

        /// Only relevant for MaxEntropyGuesser. If present, the best guess will be chosen based on
        /// best information gain after 1 guess instead of 2
        #[arg(short, long, default_value_t = false)]
        l1_entropy: bool,

        /// Only relevant for MinExpectedScoreGuesser. 
        /// Only the top_k guesses by entropy will be considered in each step.
        #[arg(short, long, default_value_t = 20)]
        top_k: usize,
    },
}

#[derive(ValueEnum, Debug, Clone)]
#[value()]
enum GuesserTypeEnum {
    MaxEntropy,
    MinExpectedScore,
}

// TODO: cli options/control over feedback pattern precomputation. e.g. if it is not found, ask if
// it should be computed, or at least tell the user to first run precomputation

fn play(
    guesser_arg: GuesserTypeEnum,
    initial_guess_arg: Option<String>,
    solution_arg: Option<String>,
    top_k: usize,
    load_feedback_patterns: bool,
) {
    let wordle = Wordle::new();

    let solution = if let Some(solution_str) = solution_arg {
        find_string_in_wordlist(solution_str, wordle.wordlist()).expect(
            "'{solution_str}' is not in the wordlist, choose a different word as the solution",
        )
    } else {
        let sol = wordle.random_word().expect("unable to choose random word");
        println!("Choosing random word as the solution: '{sol}'");
        sol
    };

    match guesser_arg {
        GuesserTypeEnum::MaxEntropy => {
            let guesser = setup_max_entropy_guesser(initial_guess_arg, true, load_feedback_patterns);

            if let Some(guesses) = wordle.play(solution, guesser) {
                println!("Solved wordle in {guesses} guesses!");
            } else {
                println!("Guesser did not find a solution!");
            };
        }
        GuesserTypeEnum::MinExpectedScore => {
            let guesser = setup_min_expected_score_guesser(initial_guess_arg, true, top_k, load_feedback_patterns);

            let start = Instant::now();
            if let Some(guesses) = wordle.play(solution, guesser) {
                println!("Solved wordle in {guesses} guesses!");
            } else {
                println!("Guesser did not find a solution!");
            };
            let duration = start.elapsed().as_secs_f32();
            println!("took {duration:.2}s");
        }
    };
}

// TODO: implement good benchmark for comparing guesser performance
// number of guesses and runtime

fn setup_max_entropy_guesser(initial_guess: Option<String>, verbose: bool, load_feedback_patterns: bool) -> MaxEntropyGuesser {
    let initial_guess = if let Some(guess) = initial_guess {
        find_string_in_wordlist(guess, Wordle::new().wordlist()).unwrap()
    } else {
        load_initial_guess(GuesserTypeEnum::MaxEntropy).expect("Unable to load precomputed initial guess. Either provide an initial guess manually or precompute it first.").get_string()
    };

    let builder = if load_feedback_patterns {
        let feedback_storage = FeedbackStorage::load().expect(
            "FeedbackStorage with precomputed patterns could not be loaded. Please precompute it first",
        );
        MaxEntropyGuesser::builder().precomputed_patterns(feedback_storage)
    } else {
        MaxEntropyGuesser::builder()
    };

    builder
        .verbose(verbose)
        .initial_guess(initial_guess)
        .build()
}

fn setup_min_expected_score_guesser(
    initial_guess: Option<String>,
    verbose: bool,
    top_k: usize,
    load_feedback_patterns: bool,
) -> MinExpectedScoreGuesser {
    let initial_guess = if let Some(guess) = initial_guess {
        find_string_in_wordlist(guess, Wordle::new().wordlist()).unwrap()
    } else {
        load_initial_guess(GuesserTypeEnum::MinExpectedScore).expect("Unable to load precomputed initial guess. Either provide an initial guess manually or precompute it first.").get_string()
    };


    let builder = if load_feedback_patterns {
        let feedback_storage = FeedbackStorage::load().expect(
            "FeedbackStorage with precomputed patterns could not be loaded. Please precompute it first",
        );
        MinExpectedScoreGuesser::builder().precomputed_patterns(feedback_storage)
    } else {
        MinExpectedScoreGuesser::builder()
    };
    builder 
        .verbose(verbose)
        .initial_guess(initial_guess)
        .top_k(top_k)
        .build()
}

fn main() {
    let args = Args::parse();

    match args.command {
        Cmd::Play {
            guesser,
            initial_guess,
            solution,
            top_k,
            load_feedback_patterns,
        } => {
            play(guesser, initial_guess, solution, top_k, load_feedback_patterns);
        }
        Cmd::Solve {
            guesser,
            initial_guess,
            top_k,
            load_feedback_patterns,
        } => {
            solve(guesser, initial_guess, top_k, load_feedback_patterns);
        }
        Cmd::Benchmark {
            guesser,
            initial_guess,
            max_games,
            top_k,
        } => {
            benchmark(guesser, max_games, initial_guess, top_k);
        }
        Cmd::CustomBench {} => {
            custom_bench();
        }
        Cmd::Precompute(precompute_cmd) => match precompute_cmd {
            PrecomputeCmd::InitialGuess {
                guesser,
                l1_entropy,
                top_k,
            } => precompute_initial_guess(guesser, !l1_entropy, top_k),
            PrecomputeCmd::FeedbackPatterns {} => precompute_patterns(),
        },
    }
}

fn solve(guesser_arg: GuesserTypeEnum, initial_guess_arg: Option<String>, top_k: usize, load_feedback_patterns: bool) {
    let wordle = Wordle::new();

    match guesser_arg {
        GuesserTypeEnum::MaxEntropy => {
            let guesser = setup_max_entropy_guesser(initial_guess_arg, true, load_feedback_patterns);

            wordle.interactive_solve(guesser);
        }
        GuesserTypeEnum::MinExpectedScore => {
            // TODO: top_k
            let guesser = setup_min_expected_score_guesser(initial_guess_arg, true, top_k, load_feedback_patterns);
            wordle.interactive_solve(guesser);
        }
    };
}

fn benchmark(
    guesser_arg: Option<GuesserTypeEnum>,
    max_games: Option<usize>,
    initial_guess: Option<String>,
    top_k: usize,
) {
    let wordle = Wordle::new();

    let initial_guess = initial_guess
        .map(|s| find_string_in_wordlist(s, wordle.wordlist()))
        .unwrap();

    let make_entropy_guesser = || {
        let initial_guess = initial_guess.map(|s| String::from(s));
        setup_max_entropy_guesser(initial_guess, false, true)
    };
    // TODO: top_k
    let make_expected_score_guesser = || {
        let initial_guess = initial_guess.map(|s| String::from(s));
        setup_min_expected_score_guesser(initial_guess, false, top_k, true)
    };

    match guesser_arg {
        Some(GuesserTypeEnum::MaxEntropy) => {
            println!("Benchmarking MaxEntropyGuesser");
            wordle.benchmark(make_entropy_guesser, max_games);
        }
        Some(GuesserTypeEnum::MinExpectedScore) => {
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
    let guesser = MaxEntropyGuesser::builder()
        .precomputed_patterns(feedback_storage)
        .build();

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

fn precompute_initial_guess(guesser_arg: GuesserTypeEnum, l2_entropy: bool, top_k: usize) {
    let feedback_storage = FeedbackStorage::load()
        .expect("FeedbackStorage could not be found. Make sure it is precomputed first.");

    let start = Instant::now();
    let guess = match guesser_arg {
        GuesserTypeEnum::MaxEntropy => {
            println!(
                "Precomputing best initial guess for MaxEntropyGuesser, based on max. expected information gain after {}",
                if l2_entropy { "2 rounds" } else { "1 round" }
            );
            let guesser = MaxEntropyGuesser::builder()
                .precomputed_patterns(feedback_storage)
                .build();
            let initial_guess = guesser.compute_best_initial_guess(l2_entropy).1;
            println!("Best initial guess for MaxEntropyGuesser: {initial_guess:?}");
            initial_guess
        }
        GuesserTypeEnum::MinExpectedScore => {
            println!("Precomputing best initial guess for MinExpectedScoreGuesser");
            let guesser = MinExpectedScoreGuesser::builder()
                .top_k(top_k)
                .precomputed_patterns(feedback_storage)
                .build();
            let initial_guess = guesser.compute_best_initial_guess();
            println!("Best initial guess for MinExpectedScoreGuesser: {initial_guess:?}");
            initial_guess
        }
    };
    let duration = start.elapsed().as_secs_f32();
    println!("took {duration:.2}s");

    if let Err(e) = store_initial_guess(guess, guesser_arg) {
        println!("Error while storing guess: {e}");
    } else {
        println!("Guess stored successfully!");
    }
}


// TODO: does precomputation overwrite existing precomputed storage? it should probably

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

fn load_initial_guess(guesser_type: GuesserTypeEnum) -> Option<Guess> {
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
    let filename = format!(
        "initial_guess_{}.storage",
        guesser_type_to_string(&guesser_type)
    );
    let file_path = data_dir.join(filename);

    let content = std::fs::read_to_string(file_path).ok()?;
    let parts: Vec<&str> = content.split(',').collect();

    if parts.len() != 3 {
        return None;
    }

    let guess_str = parts[0].trim();
    let variant_type = parts[1].trim();
    let value: f64 = parts[2].trim().parse().ok()?;

    let variant = match variant_type {
        "entropy" => wordler::GuessType::Entropy {
            entropy: value,
            solution_probability: 1.0, // We don't store this, use default
        },
        "expected_score" => wordler::GuessType::ExpectedScore { score: value },
        _ => return None,
    };

    // Convert string to &'static str by finding it in the wordlist
    let wordle = Wordle::new();
    let static_guess = wordle
        .wordlist()
        .iter()
        .find(|&&word| word == guess_str)
        .copied()?;

    Some(wordler::Guess {
        guess: static_guess,
        variant,
    })
}

fn store_initial_guess(guess: Guess, guesser_type: GuesserTypeEnum) -> Result<(), std::io::Error> {
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
    std::fs::create_dir_all(&data_dir)?;

    let filename = format!(
        "initial_guess_{}.storage",
        guesser_type_to_string(&guesser_type)
    );
    let file_path = data_dir.join(filename);

    let serialized = match guess.get_variant() {
        wordler::GuessType::Entropy { entropy, .. } => {
            format!("{},entropy,{}", guess.get_string(), entropy)
        }
        wordler::GuessType::ExpectedScore { score } => {
            format!("{},expected_score,{}", guess.get_string(), score)
        }
    };

    std::fs::write(file_path, serialized)
}

fn guesser_type_to_string(guesser_type: &GuesserTypeEnum) -> &'static str {
    match guesser_type {
        GuesserTypeEnum::MaxEntropy => "max_entropy",
        GuesserTypeEnum::MinExpectedScore => "min_expected_score",
    }
}
