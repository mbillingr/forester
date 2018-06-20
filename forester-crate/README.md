# forester
A rust crate for implementing various flavors of random forests and decision trees.

The aim of this project is to provide generic functionality for working with random forests. 
It is currently in a very early development stage. There is no real API yet; everything is in flux while I'm experimenting which concepts
work best.

## Overview

This implementation of random forests is heavily inspired by [1]. In particular, models for classification, regression, and density 
estimation are provided in a unified framework based on traits.

Conceptually, the implementation consists of two main parts:

- Generic functions for working with trees and forests
- Traits that allow these functions to understand arbitrary user data


## Literature

1. A. Criminisi, J. Shotton and E. Konukoglu, "*Decision Forests for Classification, Regression, Density Estimation, 
   Manifold Learning and Semi-Supervised Learning*", Microsoft Research technical report TR-2011-114 
   ([PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/decisionForests_MSR_TR_2011_114.pdf))
