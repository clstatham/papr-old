use std::sync::{Arc, RwLock};

use petgraph::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    dsp::{
        graph_util::{GraphInput, GraphOutput},
        Processor, Signal,
    },
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
    output_cache: RwLock<FxHashMap<NodeIndex, FxHashMap<OutputName, Signal<T>>>>,
    graph_inputs: FxHashMap<InputName, NodeIndex>,
    graph_outputs: FxHashMap<OutputName, NodeIndex>,
}

impl Graph<AudioRate> {
    pub fn new(
        inputs: FxHashMap<InputName, Input<AudioRate>>,
        outputs: FxHashMap<OutputName, Output>,
    ) -> Graph<AudioRate> {
        let mut this = Self {
            digraph: DiGraph::new(),
            node_indices_by_name: FxHashMap::default(),
            output_cache: RwLock::new(FxHashMap::default()),
            graph_inputs: FxHashMap::default(),
            graph_outputs: FxHashMap::default(),
        };
        for (inp_name, inp) in inputs {
            let (an, cn) = GraphInput::create_nodes();
            let idx = this.add_node(an, &inp_name.0);
            this.node_indices_by_name
                .insert(NodeName(inp_name.0.to_owned()), idx);
            this.graph_inputs.insert(inp_name, idx);
        }
        for (out_name, out) in outputs {
            let (an, cn) = GraphOutput::create_nodes();
            let idx = this.add_node(an, &out_name.0);
            this.node_indices_by_name
                .insert(NodeName(out_name.0.to_owned()), idx);
            this.graph_outputs.insert(out_name, idx);
        }

        this
    }
}

impl Graph<ControlRate> {
    pub fn new(
        inputs: FxHashMap<InputName, Input<ControlRate>>,
        outputs: FxHashMap<OutputName, Output>,
    ) -> Graph<ControlRate> {
        let mut this = Self {
            digraph: DiGraph::new(),
            node_indices_by_name: FxHashMap::default(),
            output_cache: RwLock::new(FxHashMap::default()),
            graph_inputs: FxHashMap::default(),
            graph_outputs: FxHashMap::default(),
        };
        for (inp_name, inp) in inputs {
            let (an, cn) = GraphInput::create_nodes();
            let idx = this.add_node(cn, &inp_name.0);
            this.node_indices_by_name
                .insert(NodeName(inp_name.0.to_owned()), idx);
            this.graph_inputs.insert(inp_name, idx);
        }
        for (out_name, out) in outputs {
            let (an, cn) = GraphOutput::create_nodes();
            let idx = this.add_node(cn, &out_name.0);
            this.node_indices_by_name
                .insert(NodeName(out_name.0.to_owned()), idx);
            this.graph_outputs.insert(out_name, idx);
        }

        this
    }
}

impl<T: GraphKind> Graph<T> {
    pub fn add_node(&mut self, node: Arc<Node<T>>, name: &str) -> NodeIndex {
        let outs = node
            .outputs
            .keys()
            .map(|out_name| (out_name.to_owned(), Signal::new(0.0)))
            .collect();
        let idx = self.digraph.add_node(node);
        self.output_cache.write().unwrap().insert(idx, outs);
        self.node_indices_by_name
            .insert(NodeName(name.to_owned()), idx);
        idx
    }

    pub fn add_edge(
        &mut self,
        source: NodeIndex,
        sink: NodeIndex,
        connection: Connection,
    ) -> EdgeIndex {
        self.digraph.add_edge(source, sink, connection)
    }

    pub fn get_output_id(&self, name: &OutputName) -> Option<NodeIndex> {
        self.graph_outputs.get(name).copied()
    }

    pub fn get_input_id(&self, name: &InputName) -> Option<NodeIndex> {
        self.graph_inputs.get(name).copied()
    }

    pub fn process_graph(
        &self,
        t: Scalar,
        sample_rate: Scalar,
        inputs: &FxHashMap<InputName, Signal<T>>,
        outputs: &mut FxHashMap<OutputName, Signal<T>>,
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
                let out = {
                    self.output_cache.read().unwrap()[&edge.source()][&edge.weight().source_output]
                };
                *self.digraph[node_id]
                    .inputs_cache
                    .write()
                    .unwrap()
                    .get_mut(&edge.weight().sink_input)
                    .unwrap() = out;
            }
            let inps = self.digraph[node_id].inputs_cache.read().unwrap().clone();
            let node = &self.digraph[node_id];
            node.processor.process(
                t,
                sample_rate,
                node.sibling_node.as_ref(),
                &inps,
                self.output_cache
                    .write()
                    .unwrap()
                    .get_mut(&node_id)
                    .unwrap(),
            )
        }

        for (out_name, out) in outputs.iter_mut() {
            let node_idx = self.graph_outputs[out_name];
            *out = self.output_cache.read().unwrap()[&node_idx][&OutputName("out".to_owned())];
        }
    }
}

impl Processor<AudioRate> for Graph<AudioRate> {
    fn process(
        &self,
        t: Scalar,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<AudioRate as GraphKind>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<AudioRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<AudioRate>>,
    ) {
        self.process_graph(t, sample_rate, inputs, outputs);
    }
}

impl Processor<ControlRate> for Graph<ControlRate> {
    fn process(
        &self,
        t: Scalar,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<ControlRate as GraphKind>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<ControlRate>>,
        outputs: &mut FxHashMap<OutputName, Signal<ControlRate>>,
    ) {
        self.process_graph(t, sample_rate, inputs, outputs)
    }
}

pub trait CreateNodes {
    fn create_nodes() -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>);
}
