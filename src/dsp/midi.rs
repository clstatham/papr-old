use papr_proc_macro::node;

use crate::Scalar;

node! {
    pub struct MidiToFreq;
    (m) -> (f)

    ~ {
        f = (2.0 as Scalar).powf((m - 69.0) / 12.0) * 440.0;
    }
}
