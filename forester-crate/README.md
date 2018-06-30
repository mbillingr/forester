# forester
A rust crate for tailoring random forests and decision trees to *your* 
data set.

The aim of this project is to provide generic functionality for working
with random forests. It is currently in a very early development stage.
There is no real API yet; everything is in flux while I'm experimenting
which concepts work best.

## Overview

This implementation of random forests is heavily inspired by (1). In
particular, models for classification, regression, and density
estimation will be provided in a unified framework based on traits.

Conceptually, the crate provides two main parts:

1. A generic framework consisting of
    - Functionality for fitting and predicting trees and forests
    - Traits that allow these functions to understand arbitrary user data
2. Common building blocks for plugging into the framework
    - Split/Performance criteria (RMSE, GINI, ...)
    - Split Finding strategies (best random, CART, ...)
    - Ensemble combiners (aggregating, boosting - to be done)

## Usage

Most implementations of random forests work on tabular data, more or
less randomly selecting which feature columns to try for a particular
split. This works only with a finite set of predefined features.
However, as described in (1), random forests can work with
infinite-dimensional feature spaces. In other words, the parameter that
identifies a feature can be continuous value rather than a discrete
column index.

An example of an infinite-dimensional feature space is a feature that is
formed as the linear combination of two columns (see
`rotational_classifier` example). Which features to use and how to
interpret them strongly depends on the data, so it hardly makes sense to
provide a few arbitrary feature extraction methods. Instead, the work
of reasoning about the data is deferred to the users of the crate, who
need to implement the [`SampleDescription`][SampleDescription] and
[`TrainingData`][TrainingData] traits. These traits define how features
are parameterized and extracted from the data, how the final prediction
in tree leaves is made, how to evaluate splits, and much more...

## Examples

Examples can be found in the [repository][repo].


## Literature

1. A. Criminisi, J. Shotton and E. Konukoglu, "*Decision Forests for
   Classification, Regression, Density Estimation, Manifold Learning and
   Semi-Supervised Learning*", Microsoft Research technical report
   TR-2011-114 ([PDF][1])


[1]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/decisionForests_MSR_TR_2011_114.pdf

[repo]: https://github.com/mbillingr/forester

[SampleDescription]: https://docs.rs/forester/0.0.2/forester/data/trait.SampleDescription.html
[TrainingData]: https://docs.rs/forester/0.0.2/forester/data/trait.TrainingData.html
