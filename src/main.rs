use wordler::{Wordle, MaxEntropyGuesser};

fn main() {
    let wordle = Wordle::new();

    let solution = wordle.random_word().expect("unable to choose random word");
    println!("Choosing random word as solution: {solution}");

    let mut guesser = MaxEntropyGuesser::new();
    guesser.set_initial_guess("tares");

    if let Some(guesses) = wordle.play(solution, guesser) {
        println!("Solved wordle in {guesses} guesses!");
    } else {
        println!("Guesser did not find a solution!");
    }
}
