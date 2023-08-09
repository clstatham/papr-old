use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::{
    dsp::{AudioRate, ControlRate},
    graph::{InputName, Node, OutputName},
    node_constructor, Scalar, PI, TAU,
};

use super::{Processor, Signal};

// todo
