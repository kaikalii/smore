use std::f32::EPSILON;

pub trait Vectorize<const N: usize> {
    fn vectorize(&self) -> [f32; N];
}

pub trait Devectorize<const N: usize>: Vectorize<N> {
    fn devectorize(vector: [f32; N]) -> Self;
}

impl Vectorize<1> for f32 {
    fn vectorize(&self) -> [f32; 1] {
        [*self]
    }
}

impl Devectorize<1> for f32 {
    fn devectorize(vector: [f32; 1]) -> Self {
        vector[0]
    }
}

impl Vectorize<1> for f64 {
    fn vectorize(&self) -> [f32; 1] {
        [*self as f32]
    }
}

impl Devectorize<1> for f64 {
    fn devectorize(vector: [f32; 1]) -> Self {
        vector[0] as f64
    }
}

impl<const N: usize> Vectorize<N> for [f32; N] {
    fn vectorize(&self) -> [f32; N] {
        *self
    }
}

impl<const N: usize> Devectorize<N> for [f32; N] {
    fn devectorize(vector: [f32; N]) -> Self {
        vector
    }
}

impl<const N: usize> Vectorize<N> for [f64; N] {
    fn vectorize(&self) -> [f32; N] {
        let mut vectorized = [0.0; N];
        for (a, b) in vectorized.iter_mut().zip(self) {
            *a = *b as f32;
        }
        vectorized
    }
}

impl<const N: usize> Devectorize<N> for [f64; N] {
    fn devectorize(vector: [f32; N]) -> Self {
        let mut devectorized = [0.0; N];
        for (a, b) in devectorized.iter_mut().zip(&vector) {
            *a = *b as f64;
        }
        devectorized
    }
}

pub struct Smore<const A: usize, const B: usize> {
    mappings: Vec<([f32; A], [f32; B])>,
}

impl<const A: usize, const B: usize> Default for Smore<A, B> {
    fn default() -> Self {
        Smore::new()
    }
}

impl<const A: usize, const B: usize> Smore<A, B> {
    pub fn new() -> Self {
        Smore {
            mappings: Vec::new(),
        }
    }
    pub fn map<AT, BT>(&mut self, input: &AT, output: &BT)
    where
        AT: Vectorize<A>,
        BT: Devectorize<B>,
    {
        self.mappings.push((input.vectorize(), output.vectorize()));
    }
    pub fn eval<W>(&self, weight: W) -> Evaluator<W, A, B> {
        Evaluator {
            weight,
            smore: self,
        }
    }
}

pub struct Evaluator<'a, W, const A: usize, const B: usize> {
    weight: W,
    smore: &'a Smore<A, B>,
}

impl<'a, W, const A: usize, const B: usize> Evaluator<'a, W, A, B> {
    pub fn get<AT, BT>(&self, input: &AT) -> BT
    where
        W: WeightFn<A>,
        AT: Vectorize<A>,
        BT: Devectorize<B>,
    {
        let mut val_sum = [0.0; B];
        let mut weight_sum = 0.0;
        let input = input.vectorize();
        for (a, b) in &self.smore.mappings {
            let weight = if let Some(weight) = self.weight.weight(&input, a) {
                weight
            } else {
                return BT::devectorize(*b);
            };
            weight_sum += weight;
            let mut val = *b;
            mul_assign(&mut val, weight);
            add_assign(&mut val_sum, &val);
        }
        div_assign(&mut val_sum, weight_sum);
        BT::devectorize(val_sum)
    }
}

pub trait WeightFn<const N: usize> {
    fn weight(&self, target: &[f32; N], other: &[f32; N]) -> Option<f32>;
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Exponential(pub f32);

impl<const N: usize> WeightFn<N> for Exponential {
    fn weight(&self, target: &[f32; N], other: &[f32; N]) -> Option<f32> {
        let mut sum = 0.0;
        for (a, b) in target.iter().zip(other) {
            sum += (a - b).abs().powf(self.0);
        }
        if sum < EPSILON {
            return None;
        }
        Some(1.0 / sum)
    }
}

impl Exponential {
    pub fn find_best<const A: usize, const B: usize>(
        training: &Smore<A, B>,
        target: &Smore<A, B>,
        min: f32,
        max: f32,
        step: f32,
    ) -> Exponential {
        let mut exp = Exponential(min);
        let mut best = exp;
        let mut best_error = f32::MAX;
        while exp.0 <= max {
            let eval = training.eval(exp);
            let avg_error = target
                .mappings
                .iter()
                .filter_map(|(input, output)| Exponential(2.0).weight(output, &eval.get(input)))
                .map(|weight| 1.0 / weight)
                .sum::<f32>()
                / target.mappings.len() as f32;
            if avg_error < best_error {
                best = exp;
                best_error = avg_error;
            }
            exp.0 += step;
        }
        best
    }
}

fn add_assign<const N: usize>(a: &mut [f32; N], b: &[f32; N]) {
    for (a, b) in a.iter_mut().zip(b) {
        *a += *b;
    }
}

fn mul_assign<const N: usize>(v: &mut [f32; N], d: f32) {
    for i in v {
        *i *= d;
    }
}

fn div_assign<const N: usize>(v: &mut [f32; N], d: f32) {
    for i in v {
        *i /= d;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn xor() {
        let solutions = [
            ([0f32, 0.0], 0f32),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ];

        let mut smore = Smore::new();
        for (input, output) in &solutions {
            smore.map(input, output);
        }
        const ERROR: f32 = 0.01;
        let eval = smore.eval(Exponential(2.0));
        for (input, output) in &solutions {
            let errored_input = [input[0] + ERROR, input[1] - ERROR];
            let errored_output: f32 = eval.get(&errored_input);
            println!("{:?} -> {}", errored_input, errored_output);
            assert!((output - errored_output).abs() < ERROR);
        }
    }
    #[test]
    fn add() {
        const RESOLUTION: i32 = 100;
        const MIN: i32 = -1000;
        const MAX: i32 = 1000;
        let training = (MIN / RESOLUTION..MAX / RESOLUTION)
            .map(|i| (i * RESOLUTION) as f32)
            .flat_map(|i| {
                (MIN / RESOLUTION..MAX / RESOLUTION)
                    .map(|j| (j * RESOLUTION) as f32)
                    .map(move |j| ([i, j], (i + j)))
            });

        let mut smore = Smore::new();

        for (input, output) in training {
            smore.map(&input, &output);
        }

        let test_data = [
            ([0.5f32, 0.5], 1f32),
            ([23.0, 7.0], 30.0),
            ([91.0, 123.0], 214.0),
            ([111.0, 222.0], 333.0),
            ([-123.0, -456.0], -579.0),
            ([-123.0, 456.0], 333.0),
            ([107.0, -2.0], 105.0),
        ];
        let mut test = Smore::new();
        for (input, output) in &test_data {
            test.map(input, output);
        }

        let weight = Exponential::find_best(&smore, &test, 1.0, 5.0, 0.1);
        println!("{:?}", weight);

        let eval = smore.eval(weight);
        for (input, output) in &test_data {
            let evaled: f32 = eval.get(input);
            println!(
                "{} + {} = {} | error: {}",
                input[0],
                input[1],
                evaled,
                (*output - evaled).abs()
            );
        }
    }
}
