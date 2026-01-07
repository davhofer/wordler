use wordler::{Wordle, ParallelMaxEntropyGuesser};

fn main() {
    let wordle = Wordle::new();

    let solution = wordle.random_word().expect("unable to choose random word");
    println!("Choosing random word as solution: {solution}");

    let guesser = ParallelMaxEntropyGuesser::new();

    if let Some(guesses) = wordle.play(solution, guesser) {
        println!("Solved wordle in {guesses} guesses!");
    } else {
        println!("Guesser did not find a solution!");
    }
}
