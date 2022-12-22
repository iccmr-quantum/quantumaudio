# quantumaudio
Python package for Quantum Representations of Audio in qubit systems &amp examples

`quantumaudio` is a Python module with class implementations for building quantum circuits that encode and decode audio signals as quantum states for simulations, running on quantum hardware and for future Quantum Signal Processing algorithms for audio.

This repository is a great companion for the Quantum Representations of Sound book chapter written by Paulo Itaboraí and Eduardo R. Miranda, in which different strategies for encoding audio in quantum machines are introduced and discussed.

This package contains class implementations for generating quantum circuits from audio signals, as well as necessary pre and post processing functions. It contatins implementations for three representation schemes cited on the publication above, namely:

- QPAM - Quantum Probability Amplitude Modulation (Simple quantum superposition or "Amplitude Encoding")
- SQPAM - Single-Qubit Probability Amplitude Modulation (similar to FRQI quantum image representations)
- QSM - Quantum State Modulation (also known as FRQA in the literature)

It also contains a simple tutorial showing how the main methods work, and a set of examples for interfacing the package with a synthesis engine (SuperCollider) to integrate the circuits with classical synthesizers for musical applications.

## Dependencies

The `quantumaudio` package alone has the following dependencies:

- qiskit (the quantum programming framework)
- numpy
- matplotlib
- bitstring (for decoding purposes)
- ipython (for listening purposes)

## Instalation

This python module is distributed as a package in PyPi. It can be installed in any operating system by using the `pip install` syntax.

- Windows
```console
pip install quantumaudio
```
- Mac &amp Linux
```console
pip3 install quantumaudio
```
