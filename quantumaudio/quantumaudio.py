#!/usr/bin/env python
# coding: utf-8

# ## Quantum Representations of Audio
# ### Class Implementation




import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.result import Counts
from qiskit.tools import job_monitor
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from bitstring import BitArray
#from abc import ABC, abstractmethod
from IPython.display import display, Audio
import matplotlib.pyplot as plt




class EncodingScheme():
    def __init__(self):
        self._qa_encoders = {
            "qpam": QPAM, 
            "sq-pam": SQ_PAM,
            "qsm": QSM,
        }
    def get_encoder(self, encoder_name):
        encoder = self._qa_encoders.get(encoder_name)
        if not encoder:
            return ValueError(encoder_name)
        return encoder    





class QPAM():
    def __init__(self):
        
        self.norm = 1.
        
    def __repr__(self):
        return self.__class__.__name__
    
    def convert(self, original_audio):
        '''Converts a digital Audio
        into an array of probability amplitudes

        -original_audio: (numpy ndarray);  
        
        Returns: (numpy ndarray)
        '''
#         print('QPAM Convert')
        # Changes the amplitude range from [-1, 1] to [0, 1]
        prepared = (original_audio.copy()+1)/2
        self.norm = np.linalg.norm(prepared)
        return prepared/self.norm
        
    
    def prepare(self, digital_amplitudes, size, regnames, Print=False):
        '''Creates a qiskit QuantumCircuit that prepares
        a Quantum Audio state using QPAM 
        (Quantum Probability Amplitude Modulation) Representation

        -digital_amplitudes -> (numpy ndarray) Propbability Amplitudes
        -size -> (Tuple(int,int)) time (lsize) and amplitude (qsize) register sizes, respectively
                    Note: for QPAM, qsize is ALWAYS 0
        
        -Print -> (bool) Toggles a simple print of the prepared quantum state

        Returns: (qiskit QuantumCircuit)'''
#         print('QPAM Prepare')
        
        # QPAM only needs the time register's size .
        # It doesn't have an amplitude register, so qsize is necessarily 0
        lsize=size[0]
        
        # Creates a Quantum Circuit
        l = QuantumRegister(lsize, regnames[0])
        qpam = QuantumCircuit(l)
        
        # Value Setting Operation
        qpam.initialize(list(digital_amplitudes), l)
            
        if Print:
            for i in range(len(digital_amplitudes)):
                print('%.3f|%d>' %(digital_amplitudes[i], i), end='')
                if i<len(digital_amplitudes)-1:
                    print(' + ', end='')
                else:
                    print()
        return qpam
    
    def measure(self, qc, treg_pos=0):
        '''Appends Measurements to a QPAM audio circuit
        -qc -> (qiskit QuantumCircuit)
        -treg_pos -> (int) Position of the time register (qa.qregs method)
        '''
#         print('QPAM Measure')
        # Accesses the QuantumRegister containing the time information
        t = qc.qregs[treg_pos]
        
        ct = ClassicalRegister(t.size, 'ct')
        qc.add_register(ct)
        qc.measure(t, ct)
        
    def reconstruct(self, lsize, counts, shots, g=None):
        '''Builds a digital Audio from qiskit histogram data, 
        considering the QPAM encoding scheme
        
        -lsize -> (int) size of the time register
        -counts -> (qiskit Counts) (result.get_counts())
        -shots -> (int) Amount of measurements of the qiskit job
        -g -> (float) Gain factor (usually proportional to the original audio's norm)

        Returns: (numpy ndarray) a Digital Audio'''
        g=self.norm if g is None else g
        
#         print('QPAM Reconstruct')
        # Builds a zeroed ndarray
        da = np.zeros(2**lsize)
        
        # Assigns the respective probabilities to the array
        index = np.array([int(i, 2) for i in counts.keys()])
        da[index] = list(counts.values())

        # Renormalization, rescaling, and shifting
        return 2*g*np.sqrt(da/shots) -1
        




class SQ_PAM():
    def __init__(self):
        pass
    
    def __repr__(self):
        return self.__class__.__name__
    
    def t_x(self, qa, t, l, Print=False):
        '''Applies an X gate on the time register, 
        whenever the respective bit of a given time index is 0
        -qa -> (qiskit QuamtumCircuit)
        -t -> (int) Time index
        -l -> (qiskit QuantumRegister) Time register
        -Print -> (bool)

        Example:
        t_x(qa, 6, l)
            (a time register 'l' with (let's say) 5 qubits in 'qa' at instant 6)

        t == 6 -> 00110 applies X gates to qubits 0, 3 and 4.
        '''
        tstr=[]
        for i in range(l.size):
            tBit = (t>>i)&1
            tstr.append(tBit)
            if not tBit:
                qa.x(l[i])
        if Print:
            print('|',end='')

            for i in reversed(tstr):    
                print(i, end='')
            print('>',end='')
    
    def r2th_x(self, qa, t, a, l, q, Print=False):
        '''SQ-PAM Value-Setting operation.     
        Applies a  controlled Ry(2*theta) gate
        to the amplitude register, controlled by the 
        time register if at the respective time index
        state.
        -qa -> (qiskit QuamtumCircuit)
        -t -> (int) Time index
        -a -> (float) Angle of rotation
        -l -> (qiskit QuantumRegister) Time register
        -q -> (qiskit QuantumRegister) Amplitude Register
        -Print -> (bool)
        '''

        # Applies the necessary X gates at index t
        self.t_x(self, qa, t, l)

        # Creates a circuit with the respective multi-controlled-Ry gate
        mc_ry = QuantumCircuit()
        mc_ry.add_register(q)
        mc_ry.ry(2*a, 0)
        mc_ry = mc_ry.control(l.size)
#         mc_ry.qregs[0].name = 't'
#         # Appends the circuit to qa
#         qa += mc_ry 
        qa.append(mc_ry, [i for i in range(l.size+q.size-1, -1, -1)])

        # Prints the state
        if Print:
            print('[cos(%.3f)|0> + sin(%.3f)|1>]' %(a,a),end='')       

        # Applis the X gates again
        self.t_x(self, qa, t, l, Print)
    
    def convert(self, original_audio):
        '''Converts a digital Audio
        into an array of angles between [0, pi/2]

        -original_audio -> (numpy ndarray)  
        
        Returns: (numpy ndarray) 
        '''
        return np.arcsin(np.sqrt((original_audio+1)/2))
    
    def prepare(self, thetas, size, regnames, Print=False):
        '''Creates a qiskit QuantumCircuit that prepares
        a Quantum Audio State using
        SQ-PAM (Single-Qubit Probability Ampolitude Modulation) Representation

        -thetas -> (numpy ndarray) array of angles bettween [0, pi/2]
        -size -> (Tuple(int,int)) time (lsize) and amplitude (qsize) register sizes, respectively
            Note: for SQ-PAM, qsize is ALWAYS 1
            
        -Print -> (bool) Toggles a simple print of the prepared quantum state

        Returns: (qiskit QuantumCircuit)'''
        
        # QPAM has a single-qubit amplitude register,
        # so 'qsize' is necessarily 1
        lsize=size[0]
        # Time register
        l = QuantumRegister(lsize, regnames[0])
        # Amplitude register
        q = QuantumRegister(1, regname[1])

        # Init quantum circuit
        sq_pam = QuantumCircuit()
        sq_pam.add_register(q)
        sq_pam.add_register(l)
        
        # Hadamard Gate in the Time Register
        sq_pam.h(l) 
        
        # Value setting operations
        for i in range(len(thetas)):        
            self.r2th_x(self, sq_pam, i, thetas[i], l, q, Print)
            if  Print and i!=len(thetas)-1:
                print(' + ')
        if Print:
            print()  
        return sq_pam
    
    def measure(self, qc, treg_pos=1, areg_pos=0):
        '''Appends Measurements to a SQ-PAM audio circuit
        qa -> (qiskit QuantumCircuit)
        treg_pos -> (int) Position of the time register (qa.qregs method)
        areg_pos -> (int) Position of the amplitude coefficient register (qa.qregs method)
        '''
        # Creates classical registers for measurement
        t=qc.qregs[treg_pos]
        c=qc.qregs[areg_pos]
        
        ct = ClassicalRegister(t.size, 'ct')
        ca = ClassicalRegister(c.size, 'ca')
        qc.add_register(ca)
        qc.add_register(ct)

        # Measures the respective quantum registers
        qc.measure(t, ct)
        qc.measure(c, ca)
        
    def reconstruct(self, lsize, counts, shots, both=False, inverted=False):
        '''Builds a digital Audio from qiskit histogram data, 
        considering the SQ-PAM encoding scheme

        -lsize -> (int) size of the time register
        -counts -> (qiskit Counts) (result.get_counts())
        -shots -> (int) Amount of measurements of the qiskit job
        -both -> (bool) retrieve both Sine and Cosine amplitudes in a tuple
        -inverted -> (bool) retrieves the cosine amplitude if true 
        (which is the phase-inverted version of the SQ-PAM)

        Returns: (numpy ndarray) a Digital Audio
        '''
        
        N = 2**lsize
        
        ca = np.zeros(N)
        sa = np.zeros(N)

        for i in counts.keys():
            (bt, ba) = i.split()
            t = int(bt,2)
            a = counts[i]
            if (ba == '0'):
                ca[t] = a
            elif (ba =='1'):
                sa[t] = a


        if both:    
            return (ca, sa)
        elif inverted:
            return 2*(ca/(ca+sa))-1
        else:
            return 2*(sa/(ca+sa))-1
        





class QSM():
    def __init__(self):
        pass
    
    def __repr__(self):
        return self.__class__.__name__
    
    def t_x(self, qc, t, l, Print=False):
        '''Applies an X gate on the time register, 
        whenever the respective bit of a given time index is 0
        -qa -> (qiskit QuamtumCircuit)
        -t -> (int) Time index
        -l -> (qiskit QuantumRegister) Time register
        -Print -> (bool)

        Example:
        t_x(qa, 6, l)
            (a time register 'l' with (let's say) 5 qubits in 'qa' at instant 6)

        t == 6 -> 00110 applies X gates to qubits 0, 3 and 4.
        '''
        tstr=[]
        for i in range(l.size):
            tBit = (t>>i)&1
            tstr.append(tBit)
            if not tBit:
                qc.x(l[i])
        if Print:
            print('(x)|',end='')

            for i in reversed(tstr):    
                print(i, end='')
            print('>',end='')
        
    def omega_t(self, qa, t, a, l, q, Print=False):
        '''QSM Value-Setting operation.     
        Applies a multi-controlled Not gate
        to the amplitude register, using the 
        a respective time index state as control.

        -qa -> (qiskit QuamtumCircuit)
        -t -> (int) Time index
        -a -> (int) Quantized audio amplitude
        -l -> (qiskit QuantumRegister) Time register
        -q -> (qiskit QuantumRegister) Amplitude Register
        -Print -> (bool)
        '''
        
        # Applies the necessary not gates at index t
        self.t_x(self, qa, t, l)
        astr=[]
        # Flips a qubit everytime aBit==1
        for i in range(q.size):
            aBit = (a>>i)&1
            astr.append(aBit)
            if aBit:
                qa.mct(l, q[i])

        if Print:        
            print('|',end='')       
            for i in reversed(astr):    
                print(i, end='')
            print('>',end='')


        self.t_x(self, qa, t, l, Print)
        
    def convert(self, original_audio):
        '''for the QSM encoding scheme, this function is dummy.'''
        return original_audio
        
    def prepare(self, digital_audio, size, tregname='t', aregname='a', Print=False):
        '''Creates a qiskit QuantumCircuit that prepares
        a Quantum Audio State using
        QSM (Quantum State Modulation) Representation

        -digital_audio -> (numpy ndarray) quantized digital audio
        -size -> (Tuple(int,int)) time (lsize) and amplitude (qsize) register sizes, respectively
        -Print -> (bool) Toggles a simple print of the prepared quantum state


        Returns: (qiskit QuantumCircuit)
        '''
        lsize=size[0]
        qsize=size[1]
        # Time register
        l = QuantumRegister(lsize, regname[0])
        # Amplitude register
        q = QuantumRegister(qsize, regname[1])

        # Init quantum circuit
        qsm = QuantumCircuit()
        qsm.add_register(q)
        qsm.add_register(l)

        # Hadamard Gate in the Time Register
        qsm.h(l)

        # Value setting operations
        for i in range(len(digital_audio)):        
            self.omega_t(self, qsm, i, digital_audio[i], l, q, Print)
            if Print and i!=len(digitalAudio)-1:
                print(' + ', end='')
        if Print:
            print()  
        return qsm
    
    def measure(self, qc, treg_pos=1, areg_pos=0):
        '''Appends Measurements to a QSM audio circuit
        -qa -> (qiskit QuantumCircuit)
        -treg -> (int) Position of the time register (qa.qregs method)
        -areg -> (int) Position of the amplitude register (qa.qregs method)
        '''
        t=qc.qregs[treg_pos]
        a=qc.qregs[areg_pos]
       
        ct = ClassicalRegister(t.size, 'ct')
        ca = ClassicalRegister(a.size, 'ca')        
        qc.add_register(ca)
        qc.add_register(ct)
        
        qc.measure(t, ct)
        qc.measure(a, ca)
        
    def reconstruct(self, lsize, counts):
        '''Builds a digital Audio from qiskit histogram data, 
        considering the QSM encoding scheme

        -lsize -> (int) size of the time register
        -counts -> (qiskit Counts) (result.get_counts())

        Returns: (numpy ndarray) a Digital Audio
        '''
    
        N = 2**lsize
        da = np.zeros(N, int)

        for i in counts.keys():
            (bt, ba) = i.split()
            t = int(bt,2)
            # The BitArray function converts binary words into signed integers,
            # in oposition to the int(ba, 2) function.
            a = BitArray(bin=ba).int
            da[t] = a

        return da





class QuantumAudio():
    
    def __init__(self, encoder_name):
        
        #Loads the methods from the specified representation
        self.encoder = EncodingScheme().get_encoder(encoder_name)
        self.encoder_name = encoder_name
        #The audio part
        self.input = np.array([])
        self.converted_input = None
        self.output = np.array([])
        
        #Qiskit part
        self.circuit = QuantumCircuit()
        self.lsize = 0 #size of the time register - 'l qubits'
        self.qsize = 0 #size of the amplitude register - 'q qubits'
        
        self.shots = None
        self.job = None
        self.result = None
        self.counts = {}
        
        
    def __repr__(self):
        return self.__class__.__name__
    
    def load_input(self, inputAudio, qsize=1):
        self.lsize=1
        if len(inputAudio)>1:
            self.lsize = int(np.ceil(np.log2(len(inputAudio))))
        
        if self.encoder_name == 'qpam':
            self.qsize = 0
        elif self.encoder_name == 'sq-pam':
            self.qsize = 1
        else:
            self.qsize = qsize       

        #Zero Padding
        zp = np.zeros(2**self.lsize - len(inputAudio))
        self.input = np.concatenate((inputAudio, zp))
        
        if self.encoder_name =='qsm':
            self.input = self.input.astype(int)
        else:
            self.input = self.input.astype(float)/float(2**(qsize-1))
        
        print("For this input, the {} representation will require:\n         {} qubits for encoding time information and \n         {} qubits for encoding ampĺitude information.".format(self.encoder.__name__, self.lsize, self.qsize))
        return self
        
    
    def convert(self):
        self.converted_input = self.encoder.convert(self, self.input)
        return self
    
    def prepare(self, tregname='t', aregname='a', Print=False):
        self.convert()
        self.circuit = self.encoder.prepare(self.encoder, self.converted_input, (self.lsize, self.qsize), (tregname, aregname), Print)
        return self
    
    def measure(self, treg_pos=None, areg_pos=None):
        
        additional_args = []
        if treg_pos != None:
            additional_args += [treg_pos]
        if areg_pos != None:
            additional_args += [areg_pos]
        self.encoder.measure(self, self.circuit, *additional_args)
        return self
            
    def run(self, shots=10, backend_name='qasm_simulator', provider=Aer):        
#         print('QuantumAudio run')
        
        self.shots = shots
        backend = provider.get_backend(backend_name)
        
        if backend_name != 'qasm_simulator':
            circuit = transpile(self.circuit, backend=backend, optimization_level=3)
            
        else:
            circuit = self.circuit
            
        job = execute(circuit, backend, shots=shots)
        if backend_name != 'qasm_simulator':
            job_monitor(job)
        self.result = job.result()
        self.counts = job.result().get_counts()
        return self
    
    def reconstruct_audio(self, **additional_kwargs):
        
        additional_args = []
        
        if self.encoder_name == 'qpam' or self.encoder_name == 'sq-pam':
            additional_args += [self.shots]        

        self.output = self.encoder.reconstruct(self, self.lsize, self.counts,                                                *additional_args, **additional_kwargs)
        return self
        
    def plot_audio(self):
        
        plt.figure(figsize=(20, 3))
        plt.plot(np.zeros(2**self.lsize), '-k', ms=0.1)
        plt.plot(self.input)
#         plt.axis('off')
        plt.title('input')
        plt.show()
        plt.close()
        
        plt.figure(figsize=(20, 3))
        plt.plot(np.zeros(2**self.lsize), '-k', ms=0.1)
        plt.plot(self.output, 'r')
#         plt.axis('off')
        plt.title('output')
        plt.show()
        
    
    def listen(self, rate=44100):
        display(Audio(self.output, rate=rate))





def requantize_input(audio, bit_depth):
    
    Q = 2**bit_depth-1
    
    eps = 1e-16
    audio_shifted = ((audio+eps+1)/2)

    audio_quantized_norm = (np.rint((Q-1)*audio_shifted+1)-1)/(Q-1)
    
    audio_quantized = audio_quantized_norm*2-1

    return audio_quantized
    





