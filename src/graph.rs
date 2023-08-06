use std::sync::{Arc, RwLock};

use petgraph::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    dsp::{Processor, Signal},
    Scalar,
};

pub trait GraphKind
where
    Self: Copy,
{
    type SiblingNode;
}

#[derive(Clone, Copy)]
pub struct AudioRate;
#[derive(Clone, Copy)]
pub struct ControlRate;
impl GraphKind for AudioRate {
    type SiblingNode = Node<ControlRate>;
}
impl GraphKind for ControlRate {
    type SiblingNode = Node<AudioRate>;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::Display)]
pub struct NodeName(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::Display)]
pub struct InputName(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::Display)]
pub struct OutputName(pub String);

#[derive(Clone)]
pub struct Connection {
    pub source_output: OutputName,
    pub sink_input: InputName,
}

pub struct Input<T: GraphKind> {
    pub name: InputName,
    pub default: Signal<T>,
}

impl<T: GraphKind> Input<T> {
    pub fn new(name: &str, default: Signal<T>) -> Self {
        Self {
            name: InputName(name.to_owned()),
            default,
        }
    }
}

pub struct Output {
    pub name: OutputName,
}

#[non_exhaustive]
pub struct Node<T: GraphKind> {
    pub sibling_node: Option<Arc<T::SiblingNode>>,
    pub inputs: FxHashMap<InputName, Input<T>>,
    pub outputs: FxHashMap<OutputName, Output>,
    pub processor: Box<dyn Processor<T>>,
    inputs_cache: RwLock<FxHashMap<InputName, Signal<T>>>,
}

impl<T: GraphKind> Node<T> {
    pub fn new(
        inputs: FxHashMap<InputName, Input<T>>,
        outputs: FxHashMap<OutputName, Output>,
        processor: Box<dyn Processor<T>>,
        sibling_node: Option<Arc<T::SiblingNode>>,
    ) -> Self {
        Self {
            inputs_cache: RwLock::new(
                inputs
                    .iter()
                    .map(|(k, v)| (k.to_owned(), v.default))
                    .collect(),
            ),
            inputs,
            outputs,
            processor,
            sibling_node,
        }
    }

    pub fn cached_input(&self, inp_name: &InputName) -> Option<Signal<T>> {
        self.inputs_cache.read().unwrap().get(inp_name).copied()
    }
}

pub struct Graph<T: GraphKind> {
    digraph: DiGraph<Arc<Node<T>>, Connection>,
    node_indices_by_name: FxHashMap<NodeName, NodeIndex>,
    output_cache: FxHashMap<NodeIndex, FxHashMap<OutputName, Signal<T>>>,
    graph_inputs: FxHashMap<InputName, NodeIndex>,
    graph_outputs: FxHashMap<OutputName, NodeIndex>,
}

impl<T: GraphKind> Graph<T> {
    pub fn new() -> Graph<T> {
        Self {
            digraph: DiGraph::new(),
            node_indices_by_name: FxHashMap::default(),
            output_cache: FxHashMap::default(),
            graph_inputs: FxHashMap::default(),
            graph_outputs: FxHashMap::default(),
        }
    }

    pub fn add_node(&mut self, node: Arc<Node<T>>, name: &str) -> NodeIndex {
        let outs = node
            .outputs
            .keys()
            .map(|out_name| (out_name.to_owned(), Signal::new(0.0)))
            .collect();
        let idx = self.digraph.add_node(node);
        self.output_cache.insert(idx, outs);
        self.node_indices_by_name
            .insert(NodeName(name.to_owned()), idx);
        idx
    }

    // pub fn add_input(&mut self, name: &str) -> NodeIndex {
    //     let (an, _) = GraphInput::create_nodes();
    //     let idx = self.add_node(an, name);
    //     self.graph_inputs.insert(InputName(name.to_owned()), idx);
    //     idx
    // }

    pub fn add_edge(
        &mut self,
        source: NodeIndex,
        sink: NodeIndex,
        connection: Connection,
    ) -> EdgeIndex {
        self.digraph.add_edge(source, sink, connection)
    }

    pub fn process(
        &mut self,
        t: Scalar,
        sample_rate: Scalar,
        inputs: &FxHashMap<InputName, Signal<T>>,
        outputs: &mut FxHashMap<NodeName, FxHashMap<OutputName, Signal<T>>>,
    ) {
        if self.digraph.node_count() == 0 {
            return;
        }

        for (input_name, value) in inputs.iter() {
            let inp_idx = self.graph_inputs[input_name];
            *self.digraph[inp_idx]
                .inputs_cache
                .write()
                .unwrap()
                .get_mut(&InputName("in".to_owned()))
                .unwrap() = *value;
        }

        let mut starts = self.digraph.externals(Direction::Incoming);
        let mut bfs = Bfs::new(
            &self.digraph,
            starts
                .next()
                .expect("Graph::process(): graph has no input/source nodes"),
        );
        for node in starts {
            bfs.stack.push_back(node);
        }
        for node in self.graph_inputs.values() {
            bfs.stack.push_back(*node);
        }

        while let Some(node_id) = bfs.next(&self.digraph) {
            for edge in self.digraph.edges_directed(node_id, Direction::Incoming) {
                let out = self.output_cache[&edge.source()][&edge.weight().source_output];
                *self.digraph[node_id]
                    .inputs_cache
                    .write()
                    .unwrap()
                    .get_mut(&edge.weight().sink_input)
                    .unwrap() = out;
            }
            let inps = self.digraph[node_id].inputs_cache.read().unwrap().clone();
            let node = &mut self.digraph[node_id];
            node.processor.process(
                t,
                sample_rate,
                node.sibling_node.as_ref(),
                &inps,
                self.output_cache.get_mut(&node_id).unwrap(),
            )
        }

        for (node_name, node_outs) in outputs.iter_mut() {
            for (out_name, out) in node_outs.iter_mut() {
                *out = self.output_cache[&self.node_indices_by_name[node_name]][out_name];
            }
        }
    }
}

impl<T: GraphKind> Default for Graph<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub trait CreateNodes {
    fn create_nodes() -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>);
}
