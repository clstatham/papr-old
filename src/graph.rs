use std::sync::{Arc, RwLock};

use petgraph::prelude::*;
use rustc_hash::FxHashMap;
use tokio::sync::watch;

use crate::{
    dsp::{
        basic::{Dac, DummyC},
        AudioProcessor, AudioSignal, ControlProcessor, ControlSignal,
    },
    Scalar,
};

#[derive(Clone, Copy)]
pub struct AudioConnection {
    pub source_output_index: usize,
    pub sink_input_index: usize,
}

pub struct AudioInput {
    pub name: Option<String>,
    pub default: AudioSignal,
}

pub struct AudioOutput {
    pub name: Option<String>,
}

pub struct AudioNode {
    pub control_node: Arc<ControlNode>,
    pub inputs: Vec<AudioInput>,
    pub outputs: Vec<AudioOutput>,
    pub processor: Box<dyn AudioProcessor>,
}

pub struct AudioGraph {
    digraph: DiGraph<AudioNode, AudioConnection>,
    input_cache: FxHashMap<NodeIndex, Vec<AudioSignal>>,
    output_cache: FxHashMap<NodeIndex, Vec<AudioSignal>>,
    dac_nodes: Vec<NodeIndex>,
}

impl AudioGraph {
    pub fn new() -> Self {
        Self {
            digraph: DiGraph::new(),
            input_cache: FxHashMap::default(),
            output_cache: FxHashMap::default(),
            dac_nodes: Vec::new(),
        }
    }

    pub fn add_dac(&mut self) -> NodeIndex {
        let idx = self.digraph.add_node(AudioNode {
            inputs: vec![AudioInput {
                name: Some("in".to_owned()),
                default: AudioSignal(0 as Scalar),
            }],
            outputs: vec![AudioOutput {
                name: Some("out".to_owned()),
            }],
            control_node: Arc::new(ControlNode::new_dummy()),
            processor: Box::new(Dac),
        });
        self.input_cache
            .insert(idx, vec![AudioSignal(0 as Scalar); 1]);
        self.output_cache
            .insert(idx, vec![AudioSignal(0 as Scalar); 1]);
        self.dac_nodes.push(idx);
        idx
    }

    pub fn add_node(&mut self, node: AudioNode) -> NodeIndex {
        let n_outs = node.outputs.len();
        let ins = node.inputs.iter().map(|inp| inp.default).collect();
        let idx = self.digraph.add_node(node);
        self.input_cache.insert(idx, ins);
        self.output_cache
            .insert(idx, vec![AudioSignal(0 as Scalar); n_outs]);
        idx
    }

    pub fn add_edge(
        &mut self,
        source: NodeIndex,
        sink: NodeIndex,
        connection: AudioConnection,
    ) -> EdgeIndex {
        self.digraph.add_edge(source, sink, connection)
    }

    pub fn process_audio(
        &mut self,
        t: Scalar,
        _inputs: &[AudioSignal],
        outputs: &mut [AudioSignal],
    ) {
        if self.digraph.node_count() == 0 {
            return;
        }
        let mut starts = self.digraph.externals(petgraph::Direction::Incoming);
        let mut bfs = Bfs::new(
            &self.digraph,
            starts
                .next()
                .expect("AudioGraph::process_audio(): graph has no input/source nodes"),
        );
        for node in starts {
            bfs.stack.push_back(node);
        }
        // todo: push back audio inputs from hardware (adc)

        while let Some(node_id) = bfs.next(&self.digraph) {
            // let inputs =
            for edge in self.digraph.edges_directed(node_id, Direction::Incoming) {
                let out = self.output_cache[&edge.source()][edge.weight().source_output_index];
                self.input_cache.get_mut(&node_id).unwrap()[edge.weight().sink_input_index] = out;
            }
            let node = &mut self.digraph[node_id];
            node.processor.process_audio(
                t,
                &self.input_cache[&node_id],
                &node.control_node,
                self.output_cache.get_mut(&node_id).unwrap(),
            )
        }

        for (i, dac_idx) in self.dac_nodes.iter().enumerate() {
            outputs[i] = self.output_cache[dac_idx][0];
        }
    }
}

pub struct ControlConnection {
    pub source_output_index: usize,
    pub sink_input_index: usize,
}

pub struct ControlInput {
    pub name: Option<String>,
    rx: RwLock<Option<watch::Receiver<ControlSignal>>>,
    pub default: ControlSignal,
    pub cached_value: RwLock<ControlSignal>,
}

impl ControlInput {
    pub fn new(name: Option<&str>, default: ControlSignal) -> Self {
        Self {
            name: name.map(|s| s.to_owned()),
            rx: RwLock::new(None),
            default,
            cached_value: RwLock::new(default),
        }
    }

    pub fn connect(&self, rx: watch::Receiver<ControlSignal>) {
        *self
            .rx
            .write()
            .expect("ControlInput::connect(): couldn't acquire self.rx write lock") = Some(rx);
    }
}

#[non_exhaustive]
pub struct ControlOutput {
    pub name: Option<String>,
    pub tx: watch::Sender<ControlSignal>,
}

impl ControlOutput {
    pub fn new(name: Option<&str>) -> Self {
        let (tx, _rx) = watch::channel(ControlSignal(0 as Scalar));
        Self {
            name: name.map(|s| s.to_owned()),
            tx,
        }
    }
    pub fn subscribe(&self) -> watch::Receiver<ControlSignal> {
        self.tx.subscribe()
    }

    pub fn send(&self, value: ControlSignal) {
        self.tx.send_replace(value);
    }
}

pub struct ControlNode {
    pub inputs: Vec<ControlInput>,
    pub outputs: Vec<ControlOutput>,
    pub processor: Box<dyn ControlProcessor>,
}

impl ControlNode {
    pub fn new_dummy() -> Self {
        Self {
            inputs: vec![],
            outputs: vec![],
            processor: Box::new(DummyC),
        }
    }

    pub fn read_input(&self, index: usize) -> ControlSignal {
        *self.inputs[index]
            .cached_value
            .read()
            .expect("ControlNode::read_input(): couldn't acquire read lock on input cached_value")
    }

    pub fn process_control(&self, t: Scalar) {
        let mut inputs = Vec::new();
        for i in 0..self.inputs.len() {
            let val = self.inputs[i]
                .rx
                .read()
                .expect("ControlNode::read_input(): couldn't acquire read lock on input rx")
                .as_ref()
                .map(|rx| *rx.borrow())
                .unwrap_or(self.inputs[i].default);
            inputs.push(val);
            *self.inputs[i].cached_value.write().expect(
                "ControlNode::process_control(): couldn't acquire write lock on input cached_value",
            ) = val;
        }
        let outputs = &self.outputs;
        self.processor.process_control(t, &inputs, outputs);
    }
}

pub struct ControlGraph {
    digraph: DiGraph<Arc<ControlNode>, ControlConnection>,
}

impl ControlGraph {
    pub fn new() -> Self {
        Self {
            digraph: DiGraph::new(),
        }
    }

    pub fn add_node(&mut self, node: Arc<ControlNode>) -> NodeIndex {
        self.digraph.add_node(node)
    }

    pub fn add_edge(
        &mut self,
        source: NodeIndex,
        sink: NodeIndex,
        connection: ControlConnection,
    ) -> EdgeIndex {
        let rx = self.digraph[source].outputs[connection.source_output_index].subscribe();
        self.digraph[sink].inputs[connection.sink_input_index].connect(rx);
        self.digraph.add_edge(source, sink, connection)
    }

    pub fn process_control(&mut self, t: Scalar) {
        if self.digraph.node_count() == 0 {
            return;
        }
        let mut starts = self.digraph.externals(petgraph::Direction::Incoming);
        let mut bfs = Bfs::new(
            &self.digraph,
            starts
                .next()
                .expect("ControlGraph::process_control(): graph has no input/source nodes"),
        );
        for node in starts {
            bfs.stack.push_back(node);
        }

        while let Some(node) = bfs.next(&self.digraph) {
            self.digraph[node].process_control(t);
        }
    }
}

pub trait CreateNodes {
    fn create_nodes() -> (AudioNode, Arc<ControlNode>);
}
