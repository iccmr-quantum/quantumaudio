# quantumaudio
Quantumaudio Module: A Python package for Quantum Representations of Audio in qubit systems & examples

`quantumaudio` is a Python module with class implementations for building quantum circuits that encode and decode audio signals as quantum states. This is primarily aimed for quantum computing simulators, but it *might* also run on real quantum hardware. The main objective is to have a readily available tools for using quantum representations of audio in artistic contexts and for studying future Quantum Signal Processing algorithms for audio.

This repository is a great companion for the [Quantum Representations of Sound: From Mechanical Waves to Quantum Circuits](https://link.springer.com/chapter/10.1007/978-3-031-13909-3_10) book chapter written by Paulo V. Itaboraí and Eduardo R. Miranda, in which different strategies for encoding audio in quantum machines are introduced and discussed.

This package contains class implementations for generating quantum circuits from audio signals, as well as necessary pre and post processing functions. 

It contatins implementations for three representation algorithms cited on the publication above, namely:

- QPAM - Quantum Probability Amplitude Modulation (Simple quantum superposition or "Amplitude Encoding")
- SQPAM - Single-Qubit Probability Amplitude Modulation (similar to [FRQI](https://link.springer.com/article/10.1007/s11128-010-0177-y) quantum image representations)
- QSM - Quantum State Modulation (also known as [FRQA](https://www.researchgate.net/publication/312091720_Flexible_Representation_and_Manipulation_of_Audio_Signals_on_Quantum_Computers) in the literature)

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

This python module is distributed as a package in PyPi. It can be installed in any operating system by using `pip` in a console or terminal:

- Windows
```console
pip install quantumaudio
```
- Mac & Linux
```console
pip3 install quantumaudio
```

Optionally, you can download the latest [release](https://github.com/iccmr-quantum/quantumaudio/releases), which also contains the examples and tutorial notebooks.

Ideally, you would `pip install` the package in your own python environment and then download the examples from the releases page.

## Usage

To learn how to use this module, refer to the [tutorial](https://github.com/iccmr-quantum/quantumaudio/blob/main/tutorial_quantum_audio_module.ipynb) notebook.

Both the tutorial and supercollider examples were written as [Jupyter Notebooks](https://jupyter.org/install) that can be read inside this repo, or run in your local Jupyter Notebook server.

## Feedback and Getting help
Please open a [new issue](https://github.com/iccmr-quantum/quantumaudio/issues/new), to help improve the code. They are most welcome.

You may gain insight by learning more about [Qiskit](https://qiskit.org/learn) and [SuperCollider](https://supercollider.github.io/examples). We also strongly reccomend the reading of the [Quantum Representations of Sound](https://link.springer.com/chapter/10.1007/978-3-031-13909-3_10) book chapter for a better understanding of quantum representations of audio.

## API Reference

Most methods and functions in the module contain docstrings for better understanding the implementation. This API documentation is available and readable [here](https://quantumaudio.readthedocs.io/en/latest/).

## Contributing

Clone/Fork this repo and help contributing to the code! Pull Requests are very welcome. You can also contact the [main author](https://github.com/Itaborala) to exchange ideas (highly reccomended). Make sure the [pytests](https://github.com/iccmr-quantum/quantumaudio/blob/main/quantumaudio/test_quantumaudio.py) are working!

## Acknowledgements

This repo was created by [Paulo Itaboraí](https://itabora.space) as part of the [QuTune Project](https://iccmr-quantum.github.io/) and of his ongoing Master's research at the [Interdisciplinary Centre for Computer Music Research](cmr.soc.plymouth.ac.uk/), University of Plymouth, UK.

It is also considered to be auxiliary material of the book [Quantum Computer Music: Foundations, Methods and Advanced Concepts](https://link.springer.com/chapter/10.1007/978-3-031-13909-3) edited by [Eduardo Reck Miranda](http://neuromusic.soc.plymouth.ac.uk/), specifically, [chapter 10](https://link.springer.com/chapter/10.1007/978-3-031-13909-3_10), on Quantum Representations of Audio, by Paulo Itaboraí and Eduardo Miranda.

`quantumaudio` has an [MIT license](https://github.com/iccmr-quantum/quantumaudio/blob/main/LICENSE). If you use this code in your research or art, please cite it according to the [citation file](https://github.com/iccmr-quantum/quantumaudio/blob/main/CITATION.cff).

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
