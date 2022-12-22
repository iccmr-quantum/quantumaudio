from quantumaudio import QuantumAudio, QPAM, SQPAM, QSM
from qiskit.result.counts import Counts # type: ignore
#from qiskit.result.counts import Counts
import pytest
import numpy as np
import sys

def test_raise_encoder_name_empty_error():
    with pytest.raises(TypeError):
        qpam = QuantumAudio()

def test_raise_encoder_name_error():
    with pytest.raises(ValueError):
        qpam = QuantumAudio('foo')

def test_create_encoder():

    qpam = QuantumAudio('qpam')
    assert qpam.encoder == QPAM
    assert qpam.encoder_name == 'qpam'
    sqpam = QuantumAudio('sqpam')
    assert sqpam.encoder == SQPAM
    assert sqpam.encoder_name == 'sqpam'
    qsm = QuantumAudio('qsm')
    assert qsm.encoder == QSM
    assert qsm.encoder_name == 'qsm'

@pytest.fixture
def qpam():
    return QuantumAudio('qpam')

@pytest.fixture
def sqpam():
    return QuantumAudio('sqpam')

@pytest.fixture
def qsm():
    return QuantumAudio('qsm')

@pytest.fixture
def input_audio():
    return np.array([0., -0.25, 0.5 , 0.75,  -0.75  ,  -1.,  0.25])

@pytest.fixture
def quantized_input_audio():
    return np.array([0, -1, 2, 3, -3, -4, 1])


def test_load_input(qpam, sqpam, qsm, input_audio):
    qpam.load_input(input_audio)
    assert qpam.input.tolist() ==  [0., -0.25, 0.5 , 0.75,  -0.75  ,  -1.,  0.25 ,  0.]
    assert qpam.treg_size == 3
    assert qpam.areg_size == 0
    sqpam.load_input(input_audio)
    assert sqpam.input.tolist() ==  [0., -0.25, 0.5 , 0.75,  -0.75  ,  -1.,  0.25 ,  0.]
    assert sqpam.treg_size == 3
    assert sqpam.areg_size == 1
    qsm.load_input(input_audio)
    assert qsm.input.tolist() == [0, 0, 0, 0, 0, -1, 0, 0]
    assert qsm.treg_size == 3

def test_load_quantized_input(qpam, sqpam, qsm, input_audio, quantized_input_audio):
    qpam.load_input(quantized_input_audio, 3)
    assert qpam.input.tolist() ==  [0., -0.25, 0.5 , 0.75,  -0.75  ,  -1.,  0.25 ,  0.]
    assert qpam.treg_size == 3
    assert qpam.areg_size == 0
    sqpam.load_input(quantized_input_audio, 3)
    assert sqpam.input.tolist() ==  [0., -0.25, 0.5 , 0.75,  -0.75  ,  -1.,  0.25 ,  0.]
    assert sqpam.treg_size == 3
    assert sqpam.areg_size == 1
    qsm.load_input(quantized_input_audio, 3)
    assert qsm.input.tolist() == [0, -1, 2, 3, -3, -4, 1, 0]
    assert qsm.treg_size == 3
    assert qsm.areg_size == 3

@pytest.fixture
def qpam_loaded(qpam, quantized_input_audio):
    return qpam.load_input(quantized_input_audio, 3)

@pytest.fixture
def sqpam_loaded(sqpam, quantized_input_audio):
    return sqpam.load_input(quantized_input_audio, 3)

@pytest.fixture
def qsm_loaded(qsm, quantized_input_audio):
    return qsm.load_input(quantized_input_audio, 3)

def test_convert(qpam_loaded):
    inp = qpam_loaded.input
    qpam_loaded._convert()
    assert qpam_loaded.converted_input.tolist() != None
    assert qpam_loaded.converted_input.tolist() == (((inp+1)/2)/np.linalg.norm((inp+1)/2)).tolist()

def test_prepare(qpam_loaded):
    qpam_loaded.prepare()

def test_circuit_workflow_qpam(qpam_loaded):
    qpam_loaded.prepare().measure().run(1)

def test_circuit_workflow_sqpam(sqpam_loaded):
    sqpam_loaded.prepare().measure().run(1)

def test_circuit_workflow_qsm(qsm_loaded):
    qsm_loaded.prepare().measure().run(1)

def test_reconstruction_qpam(qpam_loaded):
    qpam_loaded._convert()
    qpam_loaded.shots = 1000
    qpam_loaded.counts = Counts({'100': 5, '001': 60, '110': 174, '111': 106, '011': 313, '000': 116, '010': 226})
    qpam_loaded.reconstruct_audio()
    assert np.sum((qpam_loaded.output - qpam_loaded.input)**2) < 0.05

def test_reconstruction_sqpam(sqpam_loaded):
    sqpam_loaded.shots = 1000
    sqpam_loaded.counts = Counts({'110 0': 50, '011 1': 114, '001 1': 51, '011 0': 8, '100 0': 106, '001 0': 76, '000 0': 57, '101 0': 114, '111 0': 67, '100 1': 13, '010 1': 100, '010 0': 44, '000 1': 58, '111 1': 60, '110 1': 82})
    sqpam_loaded.reconstruct_audio()
    assert np.sum((sqpam_loaded.output - sqpam_loaded.input)**2) < 0.1

def test_reconstruction_qsm(qsm_loaded):
    qsm_loaded.shots = 1000
    qsm_loaded.counts = Counts({'101 100': 122, '111 000': 125, '011 011': 125, '110 001': 134, '010 010': 132, '001 111': 110, '100 101': 132, '000 000': 120})
    qsm_loaded.reconstruct_audio()
    assert np.sum((qsm_loaded.output - qsm_loaded.input)) == 0

def test_bypass_workflow_qpam(qpam_loaded):
    qpam_loaded.prepare().measure().run(10000).reconstruct_audio()
    assert np.sum((qpam_loaded.output - qpam_loaded.input)**2) < 0.05

def test_bypass_workflow_sqpam(sqpam_loaded):
    sqpam_loaded.prepare().measure().run(10000).reconstruct_audio()
    assert np.sum((sqpam_loaded.output - sqpam_loaded.input)**2) < 0.1

def test_bypass_workflow_qsm(qsm_loaded):
    qsm_loaded.prepare().measure().run(10000).reconstruct_audio()
    assert np.sum((qsm_loaded.output - qsm_loaded.input)) == 0

def test_print_qpam(qpam_loaded, capfd):
    qpam_loaded.prepare(print_state=True)
    out, err = capfd.readouterr()
    assert out == '0.324|0> + 0.243|1> + 0.487|2> + 0.568|3> + 0.081|4> + 0.000|5> + 0.406|6> + 0.324|7>\n'

def test_print_sqpam(sqpam_loaded, capfd):
    sqpam_loaded.prepare(print_state=True)
    out, err = capfd.readouterr()
    assert out == '[cos(0.785)|0> + sin(0.785)|1>]|000> + \n\
[cos(0.659)|0> + sin(0.659)|1>]|001> + \n\
[cos(1.047)|0> + sin(1.047)|1>]|010> + \n\
[cos(1.209)|0> + sin(1.209)|1>]|011> + \n\
[cos(0.361)|0> + sin(0.361)|1>]|100> + \n\
[cos(0.000)|0> + sin(0.000)|1>]|101> + \n\
[cos(0.912)|0> + sin(0.912)|1>]|110> + \n\
[cos(0.785)|0> + sin(0.785)|1>]|111>\n'

def test_print_qsm(qsm_loaded, capfd):
    qsm_loaded.prepare(print_state=True)
    out, err = capfd.readouterr()
    assert out == '|000>(x)|000> + |111>(x)|001> + |010>(x)|010> + |011>(x)|011> + |101>(x)|100> + |100>(x)|101> + |001>(x)|110> + |000>(x)|111>\n'

def test_QPAM_prepare_warning(qpam_loaded):
    #qpam_loaded.prepare(size=(3, 3))
    pass
