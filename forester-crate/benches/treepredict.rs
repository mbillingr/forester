#[macro_use]
extern crate criterion;
extern crate rand;
extern crate forester;

use criterion::Criterion;
use criterion::Bencher;

use forester::d_tree::{DeterministicTree, Node};
use forester::splitters::ThresholdSplitter;
use forester::predictors::ConstMean;
use forester::features::ColumnSelect;
use forester::datasets::TupleSample;
use forester::traits::Side;
use forester::{DeterministicSplitter, LeafPredictor, Predictor as PT};

pub type Tree = DeterministicTree<Splitter, Predictor>;
pub type Splitter = ThresholdSplitter<[Sample]>;
pub type Predictor =  ConstMean<Sample>;
pub type Sample = TupleSample<ColumnSelect, [u32; 1], f64>;

fn safe_predict(tree: &Tree, x: &[u32; 1]) -> f64 {
    let mut n = 0;
    loop {
        match tree.nodes[n] {
            Node::Split{ref split, left, right} => {
                match split.split(x) {
                    Side::Left => n = left,
                    Side::Right => n = right,
                }
            }
            Node::Leaf(ref l) => {
                return l.predict(x)
            }
            Node::Invalid => panic!("Invalid node found. Tree may not be fully constructed.")
        }
    }
}


fn build_tree(max_depth: u32) -> Tree {
    let range: (u32, u32) = (0, 2u32.pow(max_depth));
    let mut tree = DeterministicTree {nodes: vec![Node::Invalid]};
    build_tree_recursive(&mut tree, max_depth, 0, range);
    tree
}

fn build_tree_recursive(tree: &mut Tree, depth: u32, node: usize, range: (u32, u32)) {
    if depth == 0 {
        tree.nodes[node] = Node::Leaf(Predictor::new(1.0));
        return
    }

    let (a, b) = range;
    let c = (a + b) / 2;

    let left = tree.nodes.len();
    let right = left + 1;
    tree.nodes.push(Node::Invalid);
    tree.nodes.push(Node::Invalid);
    tree.nodes[node] = Node::Split{split: Splitter::new(0, c), left, right};

    build_tree_recursive(tree, depth - 1, left, (a, c));
    build_tree_recursive(tree, depth - 1, right, (c, b));
}


fn bench_partition(c: &mut Criterion) {
    fn function1(b: &mut Bencher) {
        let tree = build_tree(20);
        b.iter(|| {
            tree.predict(&[5]);
        })
    }
    c.bench_function("Partition (unsafe)", function1);


    fn function2(b: &mut Bencher) {
        let tree = build_tree(20);
        b.iter(|| {
            safe_predict(&tree, &[5]);
        })
    }
    c.bench_function("Partition (safe)", function2);
}

criterion_group!(benches, bench_partition);
criterion_main!(benches);
