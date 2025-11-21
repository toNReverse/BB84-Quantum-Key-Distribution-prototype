# Implementation of the BB84 Protocol con Noisy simulator


from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import random

def get_noise_model(prob_error):
    noise_model = NoiseModel()
    # Creiamo un errore di depolarizzazione (il qubit perde informazione)
    # Questo simula il rumore che colpisce le porte logiche o la trasmissione
    error = depolarizing_error(prob_error, 1)
    # Aggiungiamo questo errore alle porte base (H, X) e alla Misura
    noise_model.add_all_qubit_quantum_error(error, ['h', 'x', 'measure'])
    return noise_model
 
# simulate a quantum circuit and returns the measurement
def simulate(qc, noise_model=None   ):
    simulator = AerSimulator(noise_model=noise_model)
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=1024).result()
    counts = result.get_counts()
    return int(list(counts.keys())[0])


def randbit():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return simulate(qc)


def random_sequence(n):
    s = []
    for i in range(n):
        s.append(randbit())
    return s


def random_bases(n):
    b = []
    s = random_sequence(n)
    for i in range(n):
        if (s[i] == 1):
            b.append("Z")
        else:
            b.append("X")
    return b


def encode_message(bits, bases):
    message = []
    for i in range(len(bits)):
        qc = QuantumCircuit(1, 1)
        if (bits[i] == 1):
            qc.x(0)
        if (bases[i] == "X"):
            qc.h(0)
        qc.barrier()
        message.append(qc)
    return message


def measure_message(message, bases):
    n = len(message)
    for i in range(n):
        qc = message[i]
        if (bases[i] == "X"):
            qc.h(0)
        qc.barrier()
        qc.measure(0, 0)
        message[i] = qc
    return message


def decode_message(message, bases, noise_model=None):
    bits = []
    measure_message(message, bases)
    for i in range(len(message)):
        qc = message[i]
        bits.append(simulate(qc, noise_model))
    return bits


def getSampleIdx(n):
    idx = list(range(n))
    random.shuffle(idx)
    idx = idx[:int(n / 2)]
    return idx


def BB84(n, noise_level=0.0, intercept=False, verbose=True):
    #noise_level (0.0 = no rumore, 0.1 = 10% rumore)
    noise_model = None
    if noise_level > 0:
        noise_model = get_noise_model(noise_level)
        if verbose: print(f"Rumore al {noise_level * 100}%")
    # Alice sceglie n bits in modo casuale
    alice_bits = random_sequence(n)
    if verbose:
        print("Alice's Bits:")
        print(alice_bits)
    # Alice sceglie n Basi in modo casuale
    alice_bases = random_bases(n)
    if verbose:
        print("Alice's Bases:")
        print(alice_bases)
    # Alice prepara il messaggio
    message = encode_message(alice_bits, alice_bases)
    # Alice invia il messaggio a Bob

    if (intercept):
        # avviene una intercettazione
        eve_bases = ["Z"] * n
        measure_message(message, eve_bases)

    # Bob sceglie n Basi in modo casuale
    bob_bases = random_bases(n)
    if verbose:
        print("Bob's Bases:")
        print(bob_bases)
    # Bob effettua la misurazione degli n qubits
    bob_bits = decode_message(message, bob_bases, noise_model=noise_model)
    if verbose:
        print("Bob's Bits:")
        print(bob_bits)

    # alice e bob si scambiano le basi e le conforntano
    match = [i for i in range(n) if bob_bases[i] == alice_bases[i]]
    # Alice e Bob creano le loro chiavi segrete
    alice_key = [str(alice_bits[i]) for i in match]
    bob_key = [str(bob_bits[i]) for i in match]
    if verbose:
        print("Chiave di Alice:")
        print("".join(alice_key))
        print("Chiave di Bob:")
        print("".join(bob_key))

    sample = getSampleIdx(len(bob_key))
    alice_sample = [alice_key[i] for i in sample]
    bob_sample = [bob_key[i] for i in sample]
    if (alice_sample == bob_sample):
        # nessuna intercettazione
        if verbose:
            print("Chiave valida:")
        alice_key = [alice_key[i] for i in range(len(alice_key)) if i not in sample]
        # print(alice_key)
        bob_key = [bob_key[i] for i in range(len(bob_key)) if i not in sample]
        # print(bob_key)
        if (alice_key == bob_key):
            if verbose:
                print("".join(bob_key))
                print("Lunghezza della chiave: " + str(len(bob_key)))
            return 0
        else:
            if verbose:
                print("Errore")
            return 1
    else:
        if verbose:
            print("Attenzione, messaggio intercettato. Chiave non è sicura")
        return 0


def testBB84(n, run):
    errors = 0
    for i in range(run):
        errors = errors + BB84(n, 0.0,True, False)
    print("Il protocollo ha fallito nel " + str(errors * 100 / run) + "% dei casi")


n_bit = 20
n_run = 4
BB84(n = n_bit, noise_level=0.0, intercept=False, verbose=True)
testBB84(n=n_bit, run=n_run)