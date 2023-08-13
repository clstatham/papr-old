# PAPR

**P**APR **A**udio **P**rocessor **R**untime is an application for creating real-time signal processors for both audio-rate and control-rate signals with a DSP scripting language.

## Prerequisites

In order to use PAPR, you currently are required to have JACK Audio Connection Kit installed and running.

## How To Use

Open your favorite text editor, and type/paste the following:

```papr
graph main {
    || -> |@dac0|
    ~ {
        @dac0 = @sineosc(0.1 440.0 0 0);
    }
}
```

Now open a terminal and run: `papr ./hello.papr`

You should hear a sine tone!

### Example Walkthrough

`graph` indicates a new audio- or control-rate graph definition (or *graphdef*).

`main` is the name of our new graphdef. (Every script run on the command line MUST have a "main" graphdef!)

`|| -> |@dac0|` specifies the *signature* of our graphdef.

- `||` means we have no inputs.
- `|@dac0|` means that we have one input, named `@dac0`.
  - The `@` indicates that this is an audio-rate output.
  - Any output with `dac` in its name will be interpreted as something you will hear out of your speakers, and connected accordingly.

`~ {` indicates the beginning of a list of *statements*, which are typically just connections from a number of inputs to a number of outputs.

`@dac0 = @sineosc(0.1 440.0 0 0);` is a statement that connects the output of a sine wave osscilator to the output `@dac0`.

- Again, the `@` indicates that the sine oscillator will operate at audio-rate.
- The four numbers in parentheses are simply the *inputs* to the sine oscillator. In this particular case, they correspond to the amplitude, frequency, frequency-modulation amount, and frequency-modulation inputs.

## License

Licensed under dual Apache 2.0 / MIT license. See [LICENSE-APACHE.txt](LICENSE-APACHE.txt) and [LICENSE-MIT.txt](LICENSE-MIT.txt) for more information.
