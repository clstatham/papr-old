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

#[derive(Clone)]
pub struct AudioConnection {
    pub source_output: String,
    pub sink_input: String,
}

pub struct AudioInput {
    pub name: String,
    pub default: AudioSignal,
}

pub struct AudioOutput {
    pub name: String,
}

pub struct AudioNode {
    pub control_node: Arc<ControlNode>,
    pub inputs: FxHashMap<String, AudioInput>,
    pub outputs: FxHashMap<String, AudioOutput>,
    pub processor: Box<dyn AudioProcessor>,
}

pub struct AudioGraph {
    digraph: DiGraph<AudioNode, AudioConnection>,
    input_cache: FxHashMap<NodeIndex, FxHashMap<String, AudioSignal>>,
    output_cache: FxHashMap<NodeIndex, FxHashMap<String, AudioSignal>>,
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
            inputs: FxHashMap::from_iter(
                [(
                    "in".to_owned(),
                    AudioInput {
                        name: "in".to_owned(),
                        default: AudioSignal(0 as Scalar),
                    },
                )]
                .into_iter(),
            ),
            outputs: FxHashMap::from_iter(
                [(
                    "out".to_owned(),
                    AudioOutput {
                        name: "out".to_owned(),
                    },
                )]
                .into_iter(),
            ),
            control_node: Arc::new(ControlNode::new_dummy()),
            processor: Box::new(Dac),
        });
        self.input_cache.insert(
            idx,
            self.digraph[idx]
                .inputs
                .iter()
                .map(|inp| (inp.0.to_owned(), inp.1.default))
                .collect(),
        );
        self.output_cache.insert(
            idx,
            self.digraph[idx]
                .outputs
                .iter()
                .map(|out| (out.0.to_owned(), AudioSignal(0.0)))
                .collect(),
        );
        self.dac_nodes.push(idx);
        idx
    }

    pub fn add_node(&mut self, node: AudioNode) -> NodeIndex {
        let outs = node
            .outputs
            .keys()
            .map(|out_name| (out_name.to_owned(), AudioSignal(0.0)))
            .collect();
        let ins = node
            .inputs
            .iter()
            .map(|(inp_name, inp)| (inp_name.to_owned(), inp.default))
            .collect();
        let idx = self.digraph.add_node(node);
        self.input_cache.insert(idx, ins);
        self.output_cache.insert(idx, outs);
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
        sample_rate: Scalar,
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
                let out = self.output_cache[&edge.source()][&edge.weight().source_output];
                *self
                    .input_cache
                    .get_mut(&node_id)
                    .unwrap()
                    .get_mut(&edge.weight().sink_input)
                    .unwrap() = out;
            }
            let node = &mut self.digraph[node_id];
            node.processor.process_audio(
                t,
                sample_rate,
                &self.input_cache[&node_id],
                &node.control_node,
                self.output_cache.get_mut(&node_id).unwrap(),
            )
        }

        for (i, dac_idx) in self.dac_nodes.iter().enumerate() {
            outputs[i] = self.output_cache[dac_idx]["out"];
        }
    }
}

impl Default for AudioGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct ControlConnection {
    pub source_output: String,
    pub sink_input: String,
}

#[non_exhaustive]
pub struct ControlInput {
    pub name: String,
    rx: RwLock<Option<watch::Receiver<ControlSignal>>>,
    pub default: ControlSignal,
    pub cached_value: RwLock<ControlSignal>,
}

impl ControlInput {
    pub fn new(name: &str, default: ControlSignal) -> Self {
        Self {
            name: name.to_owned(),
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
    pub name: String,
    pub tx: watch::Sender<ControlSignal>,
}

impl ControlOutput {
    pub fn new(name: &str) -> Self {
        let (tx, _rx) = watch::channel(ControlSignal(0 as Scalar));
        Self {
            name: name.to_owned(),
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
    pub inputs: FxHashMap<String, ControlInput>,
    pub outputs: FxHashMap<String, ControlOutput>,
    pub processor: Box<dyn ControlProcessor>,
}

impl ControlNode {
    pub fn new_dummy() -> Self {
        Self {
            inputs: FxHashMap::default(),
            outputs: FxHashMap::default(),
            processor: Box::new(DummyC),
        }
    }

    pub fn read_input(&self, name: &str) -> ControlSignal {
        *self.inputs[name]
            .cached_value
            .read()
            .expect("ControlNode::read_input(): couldn't acquire read lock on input cached_value")
    }

    pub fn process_control(&self, t: Scalar, control_rate: Scalar) {
        let mut inputs = FxHashMap::default();
        for i in self.inputs.keys() {
            let val = self.inputs[i]
                .rx
                .read()
                .expect("ControlNode::read_input(): couldn't acquire read lock on input rx")
                .as_ref()
                .map(|rx| *rx.borrow())
                .unwrap_or(self.inputs[i].default);
            inputs.insert(i.to_owned(), val);
            *self.inputs[i].cached_value.write().expect(
                "ControlNode::process_control(): couldn't acquire write lock on input cached_value",
            ) = val;
        }
        let outputs = &self.outputs;
        self.processor
            .process_control(t, control_rate, &inputs, outputs);
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
        let rx = self.digraph[source].outputs[&connection.source_output].subscribe();
        self.digraph[sink].inputs[&connection.sink_input].connect(rx);
        self.digraph.add_edge(source, sink, connection)
    }

    pub fn process_control(&mut self, t: Scalar, control_rate: Scalar) {
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
            self.digraph[node].process_control(t, control_rate);
        }
    }
}

impl Default for ControlGraph {
    fn default() -> Self {
        Self::new()
    }
}

pub trait CreateNodes {
    fn create_nodes() -> (AudioNode, Arc<ControlNode>);
}
