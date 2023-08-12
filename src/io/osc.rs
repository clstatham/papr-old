//! based mainly on https://github.com/karnpapon/oscd/blob/main/src/osc/recv.rs
//! UNFINISHED!!!!!!!!!!!!!!

use std::{
    error::Error,
    net::{Ipv4Addr, SocketAddrV4, ToSocketAddrs, UdpSocket},
    sync::{Arc, Mutex, RwLock},
};

use rosc::{decoder::decode_udp, OscMessage, OscPacket};

use crate::{
    dsp::{Processor, Signal},
    graph::{Node, NodeName, Output},
    Scalar,
};

const POLYPHONY: usize = 1;

pub fn unfold_packet(packet: &OscPacket) -> Vec<&OscMessage> {
    let mut msgs = vec![];
    match packet {
        OscPacket::Message(msg) => msgs.push(msg),
        OscPacket::Bundle(bund) => {
            for packet in bund.content.iter() {
                msgs.extend_from_slice(&unfold_packet(packet));
            }
        }
    }
    msgs
}

pub struct OscReceiver {
    buffer: Mutex<Vec<u8>>,
    socket: UdpSocket,
    recent_packet: Mutex<Option<OscPacket>>,
    notes: [Option<(u8, u8)>; POLYPHONY],
}

impl OscReceiver {
    pub fn bind_to<A>(addr: A) -> Result<Self, std::io::Error>
    where
        A: ToSocketAddrs,
    {
        let buffer = Mutex::new(vec![0; rosc::decoder::MTU]);
        let socket = UdpSocket::bind(addr)?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            buffer,
            socket,
            recent_packet: Mutex::new(None),
            notes: [None; POLYPHONY],
        })
    }

    pub fn bind(port: u16) -> Result<Self, std::io::Error> {
        Self::bind_to(SocketAddrV4::new(Ipv4Addr::new(0, 0, 0, 0), port))
    }

    pub fn connect<A>(self, addr: A) -> Result<Self, std::io::Error>
    where
        A: ToSocketAddrs,
    {
        let OscReceiver { buffer, socket, .. } = self;
        let mut addrs = addr.to_socket_addrs()?;
        let addr = addrs.next().unwrap();
        socket.connect(addr)?;
        Ok(Self {
            buffer,
            socket,
            recent_packet: Mutex::new(None),
            notes: [None; POLYPHONY],
        })
    }

    pub fn try_recv(&self) -> Result<Option<OscPacket>, Box<dyn Error + '_>> {
        let mut buf = self.buffer.lock()?;
        let len = match self.socket.recv_from(&mut buf) {
            Ok((len, addr)) => {
                // dbg!(len, addr);
                len
            }
            Err(e) => {
                // dbg!(e);
                return Ok(None);
            }
        };
        let (_, packet) = decode_udp(&buf[..len])?;
        Ok(Some(packet))
    }
}

impl Processor for OscReceiver {
    fn process_sample(
        &mut self,
        buffer_idx: usize,
        sample_rate: crate::Scalar,
        sibling_node: Option<&Arc<<AudioRate as crate::dsp::SignalRate>::SiblingNode>>,
        inputs: &[crate::dsp::Signal],
        outputs: &mut [crate::dsp::Signal],
    ) {
    }
}

impl Processor for OscReceiver {
    fn process_sample(
        &mut self,
        buffer_idx: usize,
        sample_rate: crate::Scalar,
        sibling_node: Option<&Arc<<ControlRate as crate::dsp::SignalRate>::SiblingNode>>,
        inputs: &[crate::dsp::Signal],
        outputs: &mut [crate::dsp::Signal],
    ) {
        let mut recent_packet = self.recent_packet.lock().unwrap();
        {
            let res = self.try_recv();
            if let Ok(Some(packet)) = res {
                *recent_packet = Some(packet);
            } else {
                *recent_packet = None;
            };
        }

        if let Some(packet) = recent_packet.clone() {
            let msgs = unfold_packet(&packet);
            for msg in msgs {
                dbg!(&msg);
                if msg.addr == "/papr/midi" {
                    for arg in msg.args.iter() {
                        #[allow(
                            clippy::single_match,
                            clippy::collapsible_match,
                            clippy::match_like_matches_macro
                        )]
                        match arg {
                            rosc::OscType::Midi(midi) => {
                                let bytes = vec![midi.status, midi.data1, midi.data2];
                                let event = midly::live::LiveEvent::parse(&bytes)
                                    .expect("Error parsing MIDI event");
                                match event {
                                    midly::live::LiveEvent::Midi {
                                        channel: _,
                                        message,
                                    } => match message {
                                        midly::MidiMessage::NoteOn { key, vel } => {
                                            if let Some(idx) =
                                                self.notes.iter_mut().find_map(|idx| {
                                                    if idx.is_none() {
                                                        Some(idx)
                                                    } else {
                                                        None
                                                    }
                                                })
                                            {
                                                *idx = Some((key.as_int(), vel.as_int()));
                                            }
                                        }
                                        midly::MidiMessage::NoteOff { key, vel } => {
                                            if let Some(idx) = self.notes.iter_mut().find(|idx| {
                                                if let Some((k, v)) = idx {
                                                    *k == key.as_int() && *v == vel.as_int()
                                                } else {
                                                    false
                                                }
                                            }) {
                                                *idx = None;
                                            }
                                        }
                                        _ => {}
                                    },
                                    _ => {}
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        for i in 0..POLYPHONY {
            let (key, _vel) = self.notes[i].unwrap_or((0, 0));
            outputs[i] = Signal::new(key as Scalar);
        }
    }
}

impl OscReceiver {
    pub fn create_node<A>(
        name: &str,
        audio_buffer_len: usize,
        bind_port: u16,
        addr: A,
    ) -> (Arc<Node>, Arc<Node>)
    where
        A: ToSocketAddrs,
    {
        let this = Arc::new(RwLock::new(Self::bind(bind_port).unwrap()));
        let cn = Arc::new(Node::new(
            NodeName::new(name),
            1,
            vec![],
            (0..POLYPHONY)
                .map(|i| Output {
                    name: format!("note{}", i),
                })
                .collect(),
            crate::graph::ProcessorType::Builtin(this.clone()),
            None,
        ));
        let an = Arc::new(Node::new(
            NodeName::new(name),
            audio_buffer_len,
            vec![],
            vec![],
            crate::graph::ProcessorType::Builtin(this),
            Some(cn.clone()),
        ));
        (an, cn)
    }
}
