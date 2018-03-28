use std::cmp;
use std::f64;
use std::iter;
use std::marker::PhantomData;
use std::ops;
use std::slice;

//use rand::thread_rng;

/// The side of a split
enum Side {
    Left,
    Right,
}

/// Prediction of the final Leaf value.
trait LeafPredictor {
    type X;
    type Y;

    /// predicted value
    fn predict(&self, x: &[Self::X]) -> Self::Y;

    /// fit predictor to data
    fn fit(data: &DataSubset<Self::X, Self::Y>) -> Self;
}

/// The probabilistic leaf predictor models uncertainty in the prediction.
trait ProbabilisticLeafPredictor: LeafPredictor {
    /// probability of given output `p(y|x)`
    fn prob(&self, x: &[Self::X], y: Self::Y) -> f64;
}

/// Extract feature from sample.
trait FeatureExtractor {
    type X: IntoIterator;
    type F;
    fn new_random(x: &Self::X) -> Self;
    fn extract(&self, x: &Self::X) -> Self::F;
}

/// Splits data at a tree node. This is a marker trait, shared by more specialized Splitters.
trait Splitter  {
    type X: IntoIterator;

    fn new_random(x: &Self::X) -> Self;
}

/// Assigns a sample to either side of the split.
trait DeterministicSplitter: Splitter {
    fn split(&self, x: &[Self::X]) -> Side;
}

/// Assigns a sample to both sides of the split with some probability each.
trait ProbabilisticSplitter: Splitter {
    /// Probability that the sample belongs to the left side of the split
    fn p_left(&self, x: &[Self::X]) -> f64;

    /// Probability that the sample belongs to the right side of the split
    fn p_right(&self, x: &[Self::X]) -> f64 {1.0 - self.p_left(x)}
}

/// Linear regression with intercept
struct LinearRegression<T> {
    intercept: T,
    weights: Vec<T>,
}

impl<T> LeafPredictor for LinearRegression<T>
    where T: ops::Mul<Output=T> + ops::AddAssign + Copy
{
    type X = T;
    type Y = T;

    fn predict(&self, x: &[Self::X]) -> Self::Y {
        let mut result = self.intercept;
        for (xi, wi) in x.iter().zip(&self.weights) {
            result += *xi * *wi;
        }
        result
    }

    fn fit(data: &DataSubset<T, T>) -> Self {
        unimplemented!("LinearRegression::fit()")
    }
}

#[derive(Debug)]
struct ConstGaussian<T> {
    mean: f64,
    variance: f64,

    _p: PhantomData<T>,
}

impl<T> LeafPredictor for ConstGaussian<T> {
    type X = T;
    type Y = f64;

    fn predict(&self, x: &[Self::X]) -> Self::Y {
        self.mean
    }

    fn fit(data: &DataSubset<T, f64>) -> Self {
        let mean = data.i.iter()
            .map(|&i| data.y[i]).sum::<f64>() / data.i.len() as f64;
        let variance = data.i.iter()
            .map(|&i| (data.y[i] - mean))
            .map(|yi| yi * yi).sum::<f64>() / data.i.len() as f64;
        ConstGaussian {
            mean,
            variance,
            _p: PhantomData,
        }
    }

}

impl<T> ProbabilisticLeafPredictor for ConstGaussian<T> {
    fn prob(&self, x: &[Self::X], y: Self::Y) -> f64 {
        f64::exp(-(y - self.mean).powi(2) / (2.0 * self.variance)) / (2.0 * f64::consts::PI * self.variance).sqrt()
    }
}

/// Defines a feature simply as a column in the sample.
#[derive(Debug)]
struct SelectFeature<T> {
    col: usize,
    _p: PhantomData<T>,
}

impl<T> FeatureExtractor for SelectFeature<T>
    where T: ops::Mul<Output=T> + ops::Add<Output=T> + Copy
{
    type X = T;
    type F = T;

    fn new_random(x: &Self::X) -> Self {
        unimplemented!("SelectFeature::new_random()")
    }

    fn extract(&self, x: &Self::X) -> Self::F {
        x[self.col]
    }
}

/// Defines a feature as the linear combination of two sample columns.
#[derive(Debug)]
struct Rot2DFeature<T> {
    ia: usize,
    ib: usize,
    wa: T,
    wb: T,
}

impl<T> FeatureExtractor for Rot2DFeature<T>
    where T: ops::Mul<Output=T> + ops::Add<Output=T> + Copy
{
    type X = T;
    type F = T;

    fn new_random(x: &[Self::X]) -> Self {
        unimplemented!("Rot2DFeature::new_random()")
    }

    fn extract(&self, x: &[Self::X]) -> Self::F {
        x[self.ia] * self.wa + x[self.ib] * self.wb
    }

}

/// Use a simple threshold for deterministic split.
#[derive(Debug)]
struct ThresholdNode<F: FeatureExtractor> {
    threshold: F::F,
    feature: F,
}

impl<F: FeatureExtractor> Splitter for ThresholdNode<F> {
    type X=F::X;

    fn new_random(x: &Self::X) -> Self
    {
        unimplemented!()
        /*let feature = F::new_random(x]);
        ThresholdNode {
            threshold,
            feature
        }*/
    }
}

impl<F: FeatureExtractor> DeterministicSplitter for ThresholdNode<F>
    where F::F: cmp::PartialOrd
{
    fn split(&self, x: &[F::X]) -> Side {
        let f = self.feature.extract(x);
        if f <= self.threshold {
            Side::Left
        } else {
            Side::Right
        }
    }
}

/// A decision tree node. Can be either a split node with a Splitter and two children, or a leaf
/// node with a LeafPredictor.
#[derive(Debug)]
enum Node<S: Splitter, L: LeafPredictor>
{
    Split(S, usize, usize),
    Leaf(L),
    Invalid,  // placeholder used during tree contstruction
}

/// Generic decision tree.
#[derive(Debug)]
struct Tree<S: Splitter, P: LeafPredictor<X=S::X>>
{
    nodes: Vec<Node<S, P>>,
}

impl<S: Splitter, P: LeafPredictor<X=S::X>> Tree<S, P> {
    fn split_node(&mut self, n: usize, split: S) -> (usize, usize) {
        let l = self.nodes.len();
        let r = l + 1;
        self.nodes.push(Node::Invalid);
        self.nodes.push(Node::Invalid);
        self.nodes[n] = Node::Split(split, l, r);
        (l, r)
    }
}

impl<S: DeterministicSplitter, P: LeafPredictor<X=S::X>> Tree<S, P> {
    /// Pass a sample `x` down the tree and predict output of final leaf.
    fn predict(&self, x: &[S::X]) -> P::Y {
        let mut n = 0;
        loop {
            match self.nodes[n] {
                Node::Split(ref s, l, r) => {
                    match s.split(x) {
                        Side::Left => n = l,
                        Side::Right => n = r,
                    }
                }
                Node::Leaf(ref l) => {
                    return l.predict(x)
                }
                Node::Invalid => panic!("Invalid node found. Tree may not be fully constructed.")
            }
        }
    }
}

#[derive(Debug)]
struct Vec2D<T> {
    data: Vec<T>,
    n_columns: usize
}

impl<T: Clone> Vec2D<T> {
    pub fn new() -> Vec2D<T> {
        Vec2D {
            data: Vec::new(),
            n_columns: 0
        }
    }

    pub fn from_slice(x: &[T], n_columns: usize) -> Vec2D<T> {
        assert_eq!(0, x.len() % n_columns);
        Vec2D {
            data: x.into(),
            n_columns
        }
    }
}

impl<'a, T> IntoIterator for &'a Vec2D<T> {
    type Item = &'a [T];
    type IntoIter = slice::Chunks<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.chunks(self.n_columns)
    }
}

impl<T> ops::Index<usize> for Vec2D<T> {
    type Output = [T];

    fn index(&self, idx: usize) -> &Self::Output {
        let a = idx * self.n_columns;
        let b = a + self.n_columns;
        &self.data[a..b]
    }
}

struct Slice2D<'a, T>
    where [T]: 'a
{
    data: &'a [T],
    n_columns: usize,
}

impl<'a, T> Slice2D<'a, T> {
    pub fn new(data: &[T], n_columns: usize) -> Slice2D<T> {
        assert_eq!(0, data.len() % n_columns);
        Slice2D {data, n_columns}
    }

    pub fn n_rows(&self) -> usize {
        self.data.len() / self.n_columns
    }

    pub fn row(&self, i: usize) -> &[T] {
        let a = i * self.n_columns;
        let b = a + self.n_columns;
        &self.data[a..b]
    }
}

impl<'a, T> Copy for Slice2D<'a, T> { }

impl<'a, T> Clone for Slice2D<'a, T> {
    fn clone(&self) -> Slice2D<'a, T> {
        Slice2D {
            data: self.data,
            n_columns: self.n_columns
        }
    }
}

struct DataSubset<'a, 'b, I:'a, O:'a> {
    x: Slice2D<'a, I>,
    y: &'a [O],
    i: &'b [usize],
}


trait SplitFitter {
    type Split: Splitter;
    type Criterion: SplitCriterion;
    fn find_split(&self,
                  x: &<Self::Split as Splitter>::X,
                  y: &<Self::Criterion as SplitCriterion>::Y)
                  -> Option<(Self::Split, Vec<usize>, Vec<usize>)>;
}


trait SplitCriterion {
    type Y;
    type C: cmp::PartialOrd + Copy;
    fn calc_presplit(y: &Self::Y) -> Self::C;
    fn calc_postsplit(yl: &Self::Y, yr: &Self::Y) -> Self::C;
}


struct VarCriterion<T> {
    _p: PhantomData<T>
}

impl<T: IntoIterator> SplitCriterion for VarCriterion<T>
    where T::Item: Into<f64> + Copy
{
    type Y=T;
    type C=f64;

    fn calc_presplit(y: &Self::Y) -> f64 {
        let mean = y.into_iter().map(|yi|yi.into()).sum::<f64>();
        y.into_iter().map(|yi| yi.into() - mean).map(|yi| yi * yi).sum()
        //let mean: Vec<f64> = y.rows().map(|yi|yi.into()).collect();
        /*let mut mean = 0.0;
        y.row_op(|yi| mean += yi.into());
        mean /= y.n_rows() as f64;*/
        //let mean: f64 = y.rows().map(|yi| yi.into()).sum::<f64>() / y.n_rows() as f64;
        //y.rows().map(|yi| yi.into() - mean).map(|yi| yi * yi).sum::<f64>() / y.n_rows() as f64
    }

    fn calc_postsplit(yl: &Self::Y, yr: &Self::Y) -> f64 {
        Self::calc_presplit(yl) + Self::calc_presplit(yr)
    }
}


struct BestRandomSplit<S, C> {
    n_splits: usize,
    _p: PhantomData<(S, C)>,
}

impl<S: DeterministicSplitter, C: SplitCriterion> SplitFitter for BestRandomSplit<S, C> {
    type Split = S;
    type Criterion = C;
    fn find_split(&self, x: &S::X, y: &C::Y) -> Option<(Self::Split, Vec<usize>, Vec<usize>)>
    {
        let mut best_criterion = None;
        let mut best_split = None;

        let parent_criterion = C::calc_presplit(y);

        for _ in 0..self.n_splits {
            let split: S = Self::Split::new_random(x);

            let mut l = Vec::new();
            let mut r = Vec::new();

            for i in 0..y.len() {
                match split.split(x.row(i)) {
                    Side::Left => l.push(i),
                    Side::Right => r.push(i),
                }
            }

            let pc = C::calc_postsplit(y, &l, &r);

            if pc < parent_criterion {
                match best_criterion {
                    None => {
                        best_criterion = Some(pc);
                        best_split = Some((split, l, r));
                    }
                    Some(b) => if pc < b {
                        best_criterion = Some(pc);
                        best_split = Some((split, l, r));
                    }
                }
            }
        }

        best_split
    }
}


/// Generic tree fitter
struct TreeFitter<S: Splitter, C: SplitCriterion, P: LeafPredictor<X=S::X, Y=C::Y>> {
    max_samples_leaf: usize,
    split_finder: BestRandomSplit<S, C>,
    _p: PhantomData<P>
}

impl<S: DeterministicSplitter, C: SplitCriterion, P: LeafPredictor<X=S::X, Y=C::Y>> TreeFitter<S, C, P>
{
    /// fit tree to a data set
    fn fit(&self, x: Slice2D<S::X>, y: &[P::Y]) -> Tree<S, P> {
        let mut tree = Tree {
            nodes: vec![Node::Invalid]
        };

        let i: Vec<_> = (0..x.n_rows()).collect();

        self.recursive_fit(&mut tree, DataSubset{x, y, i: &i}, 0);
        tree
    }

    fn recursive_fit(&self, tree: &mut Tree<S, P>, data: DataSubset<S::X, C::Y>, n: usize)
    {
        let split;

        if data.i.len() < self.max_samples_leaf {
            split = None;
        } else {
            split = self.split_finder.find_split(&data);
        }

        match split {
            None => tree.nodes[n] = Node::Leaf(P::fit(&data)),
            Some((s, l_idx, r_idx)) => {
                let (l, r) = tree.split_node(n, s);

                self.recursive_fit(tree, DataSubset{x: data.x, y: data.y, i: &l_idx}, l);
                self.recursive_fit(tree, DataSubset{x: data.x, y: data.y, i: &r_idx}, r);
            }
        }
    }
}


pub fn test() {
    let tree: Tree<ThresholdNode<Rot2DFeature<f64>>, LinearRegression<f64>>;

    let x = vec![1, 2, 3, 4, 5, 6];

    let tree: Tree<ThresholdNode<Rot2DFeature<_>>, LinearRegression<_>> = Tree {
        nodes: vec!{Node::Leaf(LinearRegression{intercept: 0, weights: vec![1, 1, 1, 1, 1, 1]})}
    };

    let x = vec![1, 2, 3, 4, 5, 6];

    let tree: Tree<ThresholdNode<SelectFeature<_>>, LinearRegression<_>> = Tree {
        nodes: vec!{
            Node::Split(ThresholdNode{feature: SelectFeature{col: 2, _p: PhantomData}, threshold: 2}, 1, 2),
            Node::Leaf(LinearRegression{intercept: 0, weights: vec![1, 1, 1, 1, 1, 1]}),
            Node::Leaf(LinearRegression{intercept: 1, weights: vec![1, 1, 1, 1, 1, 1]})
        }
    };

    let y = tree.predict(&x);
    println!("{}", y);

    let tree: Tree<ThresholdNode<Rot2DFeature<_>>, ConstGaussian<_>> = Tree {
        nodes: vec!{Node::Leaf(ConstGaussian{mean: 42.0, variance: 1.0, _p: PhantomData})}
    };

    let y = tree.predict(&x);
    println!("{}", y);


    let x = vec![1, 2, 3, 4, 5, 6];
    let x = Slice2D::new(&x, 1);
    let y = &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let tf: TreeFitter<ThresholdNode<SelectFeature<_>>, VarCriterion<_>, ConstGaussian<_>> = TreeFitter {
        max_samples_leaf: 3,
        split_finder: BestRandomSplit{n_splits: 10, _p: PhantomData},
        _p: PhantomData
    };

    let tree = tf.fit(x, y);
    println!("{:?}", tree);
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
