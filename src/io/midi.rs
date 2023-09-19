use std::sync::{Arc, RwLock};

use midir::{MidiInput, MidiInputConnection};
use midly::live::LiveEvent;
use miette::{Diagnostic, Result};
use spin::Once;
use thiserror::Error;

use crate::{
    dsp::{Processor, Signal, SignalRate},
    graph::{Node, NodeName, Output},
    Scalar,
};

pub static MIDI_CHAN: Once<crossbeam_channel::Receiver<Vec<u8>>> = Once::INIT;

const POLYPHONY: usize = 1;

#[derive(Debug, Error, Diagnostic)]
pub enum MidiError {
    #[error("MIDI error: {0}")]
    Connect(#[from] midir::ConnectError<MidiInput>),
    #[error("MIDI error: {0}")]
    Init(#[from] midir::InitError),
    #[error("MIDI error: MIDI channel not initialized")]
    NotInitialized,
    #[error("MIDI error: {0}")]
    Crossbeam(#[from] crossbeam_channel::RecvError),
    #[error("MIDI error: {0}")]
    Midly(#[from] midly::Error),
}

pub struct MidiContext {
    _conn_in: MidiInputConnection<()>,
}

impl MidiContext {
    pub fn new(name: &str, port: Option<usize>) -> Result<Self> {
        let mut midi_in = MidiInput::new(name).map_err(MidiError::Init)?;
        midi_in.ignore(midir::Ignore::None);
        let in_ports = midi_in.ports();
        println!("Available MIDI ports:");
        for (i, port) in in_ports.iter().enumerate() {
            println!(
                "    [{i}] {}",
                midi_in.port_name(port).unwrap_or("(unknown)".to_owned())
            );
        }
        let port = port.unwrap_or(0);
        println!("Selecting port [{port}]");
        let port = &in_ports[port];

        let (tx, rx) = crossbeam_channel::unbounded();
        let _conn_in = midi_in
            .connect(
                port,
                name,
                move |_, message, _| {
                    // the midi callback is allowed to panic
                    #[allow(clippy::unwrap_used)]
                    tx.send(message.to_vec()).unwrap();
                },
                (),
            )
            .map_err(MidiError::Connect)?;
        MIDI_CHAN.call_once(|| rx);
        Ok(Self { _conn_in })
    }
}

pub struct NoteIn {
    notes: [Option<(u8, u8)>; POLYPHONY],
}

impl Processor for NoteIn {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        _inputs: &[Signal],
        outputs: &mut [Signal],
    ) -> Result<()> {
        if let Ok(msg) = MIDI_CHAN.get().ok_or(MidiError::NotInitialized)?.try_recv() {
            let msg = LiveEvent::parse(&msg).map_err(MidiError::Midly)?;
            if let LiveEvent::Midi {
                channel: _,
                message,
            } = msg
            {
                match message {
                    midly::MidiMessage::NoteOn { key, vel } => {
                        if let Some(idx) = self.notes.iter_mut().find_map(|idx| {
                            if idx.is_none() {
                                Some(idx)
                            } else {
                                None
                            }
                        }) {
                            *idx = Some((key.as_int(), vel.as_int()));
                        }
                    }
                    midly::MidiMessage::NoteOff { key, vel: _ } => {
                        if let Some(idx) = self.notes.iter_mut().find(|idx| {
                            if let Some((k, _)) = idx {
                                *k == key.as_int()
                            } else {
                                false
                            }
                        }) {
                            *idx = None;
                        }
                    }
                    _ => {}
                }
            }
        }
        for i in 0..POLYPHONY {
            let (key, vel) = self.notes[i].unwrap_or((0, 0));
            outputs[i] = Signal::Scalar(key as Scalar);
            outputs[i + POLYPHONY] = Signal::Scalar(vel as Scalar / 127.0);
        }
        Ok(())
    }
}

impl NoteIn {
    pub fn create_node(name: &str) -> Arc<Node> {
        let this = Box::new(RwLock::new(Self {
            notes: [None; POLYPHONY],
        }));

        Arc::new(Node::new(
            NodeName::new(name),
            vec![],
            (0..POLYPHONY)
                .map(|i| Output {
                    name: format!("note{}", i),
                })
                .chain((0..POLYPHONY).map(|i| Output {
                    name: format!("vel{}", i),
                }))
                .collect(),
            crate::graph::ProcessorType::Builtin(this),
        ))
    }
}
