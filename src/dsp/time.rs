use papr_proc_macro::node;

use crate::Scalar;

node! {
    pub struct Clock;
    (period, width) -> (out)

    ~ {
        if *period == 0.0 {
            out = 0.0;
        } else if t % period < period * width {
            out = 1.0;
        } else {
            out = 0.0;
        }
    }
}

node! {
    pub struct Delay {
        buf: Vec<Scalar>,
        read_head: Scalar,
        write_head: Scalar,
        delay_current: Scalar,
    }
    (input, delay) -> (out)

    ~ {
        let delay_desired_secs = *delay;

        self.delay_current =
            self.delay_current + 0.00005 * (delay_desired_secs - self.delay_current);
        // self.delay_current = delay_desired_secs;

        let sample_offset = self.delay_current * signal_rate.rate();

        self.buf[self.write_head as usize] = *input;

        // interpolate
        out = {
            let mut trunc_read = (self.read_head as usize).min(self.buf.len() - 1);
            let sample0 = self.buf[trunc_read];
            let weight_sample1 = self.read_head - (trunc_read as Scalar);

            trunc_read += 1;
            if trunc_read >= self.buf.len() {
                trunc_read = 0;
            }
            let sample1 =self. buf[trunc_read];
            sample0 + weight_sample1 * (sample1 - sample0)
        };

        self.write_head += 1.0;
        if self.write_head >=self. buf.len() as Scalar {
            self.write_head = 0.0;
        }
        self.read_head = self.write_head - sample_offset;
        if self.read_head < 0.0 {
            self.read_head += self.buf.len() as Scalar;
        }
    }
}
