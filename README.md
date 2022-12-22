# quantumaudio
Python package for Quantum Representations of Audio in qubit systems & examples

`quantumaudio` is a Python module with class implementations for building quantum circuits that encode and decode audio signals as quantum states for simulations, running on quantum hardware and for future Quantum Signal Processing algorithms for audio.

This repository is a great companion for the [Quantum Representations of Sound: From Mechanical Waves to Quantum Circuits](https://link.springer.com/chapter/10.1007/978-3-031-13909-3_10) book chapter written by Paulo V. Itaboraí and Eduardo R. Miranda, in which different strategies for encoding audio in quantum machines are introduced and discussed.

This package contains class implementations for generating quantum circuits from audio signals, as well as necessary pre and post processing functions. It contatins implementations for three representation schemes cited on the publication above, namely:

- QPAM - Quantum Probability Amplitude Modulation (Simple quantum superposition or "Amplitude Encoding")
- SQPAM - Single-Qubit Probability Amplitude Modulation (similar to FRQI quantum image representations)
- QSM - Quantum State Modulation (also known as FRQA in the literature)

There is a Jupyter Notebook [tutorial](https://github.com/iccmr-quantum/quantumaudio/blob/main/tutorial_quantum_audio_module.ipynb) showing how the main methods work and general implementation workflow with the package. Additionally, to listen the results, there is a set of [examples](https://github.com/iccmr-quantum/quantumaudio/tree/main/examples_with_supercollider) for interfacing the quantum circuits with [SuperCollider](https://supercollider.github.io/), a powerful synthesis engine for live musical applications.

## Dependencies

The `quantumaudio` package alone has the following dependencies:

- qiskit (the quantum programming framework)
- numpy
- matplotlib
- bitstring (for decoding purposes)
- ipython (for listening purposes inside jupyter notebooks)

For running the [supercollider examples](https://github.com/iccmr-quantum/quantumaudio/tree/main/examples_with_supercollider), additional packages are needed:

- SuperCollider scsynth ([install SuperCollider](https://supercollider.github.io/downloads))
- [pyliblo](https://pypi.org/project/pyliblo/)
- [python-supercollider client](https://pypi.org/project/supercollider/) (`pip install supercollider`)

## Installation

This python module is distributed as a package in PyPi. It can be installed in any operating system by using the `pip install` syntax.

- Windows
```console
pip install quantumaudio
```
- Mac &amp Linux
```console
pip3 install quantumaudio
```
