from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
import random
import numpy as np
import matplotlib.pyplot as plt


def get_noise_model(prob_error, readout_noise=False):
    noise_model = NoiseModel()

    # Errore quantistico
    if prob_error > 0:
        error = depolarizing_error(prob_error, 1)
        noise_model.add_all_qubit_quantum_error(error, ['h', 'x'])

    # Readout noise (per rendere realistico il caso no-noise / no-eve)
    if readout_noise:
        ro = ReadoutError([[0.97, 0.03],
                           [0.03, 0.97]])
        noise_model.add_all_qubit_readout_error(ro)

    return noise_model


def random_bits(n):
    return [random.randint(0,1) for _ in range(n)]

def random_bases(n):
    return [random.choice(['X','Z']) for _ in range(n)]

# ==========================
# BB84 completo
# ==========================

def BB84_single_circuit(n, noise_level=0.0, intercept=False, readout_noise=False):

    alice_bits  = random_bits(n)
    alice_bases = random_bases(n)
    qc = QuantumCircuit(n, n)

    # Codifica Alice
    for i in range(n):
        if alice_bits[i] == 1:
            qc.x(i)
        if alice_bases[i] == 'X':
            qc.h(i)

    # -------------------------
    # Intercettazione Eve corretta
    # -------------------------
    if intercept:
        eve_bases = random_bases(n)
        for i in range(n):

            # Simulazione della misura realistica
            if eve_bases[i] == alice_bases[i]:
                eve_meas = alice_bits[i]
            else:
                eve_meas = random.randint(0,1)

            # Reinvia il qubit
            qc.reset(i)
            if eve_meas == 1:
                qc.x(i)
            if eve_bases[i] == 'X':
                qc.h(i)

    # -------------------------
    # Basi Bob
    # -------------------------
    bob_bases = random_bases(n)
    for i in range(n):
        if bob_bases[i] == 'X':
            qc.h(i)
        qc.measure(i, i)

    # Simulazione
    noise_model = get_noise_model(noise_level, readout_noise=readout_noise)
    simulator = AerSimulator(noise_model=noise_model)

    result = simulator.run(qc, shots=1).result()
    measured = list(result.get_counts().keys())[0][::-1]
    bob_bits = list(map(int, measured))

    # Sifting
    matching = [i for i in range(n) if alice_bases[i] == bob_bases[i]]
    alice_key = [alice_bits[i] for i in matching]
    bob_key   = [bob_bits[i]  for i in matching]

    if len(alice_key) == 0:
        return 0, [], [], 0, qc

    errors = sum(a != b for a,b in zip(alice_key, bob_key))
    qber = errors / len(alice_key)

    return qber, alice_key, bob_key, matching, qc

# ==========================
# Esempio di esecuzione BB84
# ==========================

qber, alice_key, bob_key, matching, qc = BB84_single_circuit(
    n=20,
    noise_level=0.05,
    intercept=False,
    readout_noise=True
)

print("QBER:", qber)
print("Alice:", alice_key)
print("Bob:  ", bob_key)

# ==========================
# Parte grafica BB84 
# ==========================

def simulate_bb84_graph(n_bits=100, noise_prob=0.0, eve_prob=0.0):
    sender_bits  = [random.randint(0,1) for _ in range(n_bits)]
    sender_bases = [random.choice(['+', 'x']) for _ in range(n_bits)]
    receiver_bases = [random.choice(['+', 'x']) for _ in range(n_bits)]
    receiver_bits = []

    for i in range(n_bits):
        bit = sender_bits[i]

        # Eve
        if random.random() < eve_prob:
            eve_basis = random.choice(['+', 'x'])
            if eve_basis != sender_bases[i]:
                bit = random.randint(0,1)

        # Rumore
        if random.random() < noise_prob:
            bit = 1 - bit

        # Misura Bob
        if sender_bases[i] != receiver_bases[i]:
            bit = random.randint(0,1)

        receiver_bits.append(bit)

    matching = [i for i in range(n_bits) if sender_bases[i] == receiver_bases[i]]
    if not matching:
        return 0

    errors = sum(sender_bits[i] != receiver_bits[i] for i in matching)
    return (errors / len(matching)) * 100

# ---- Grafico 1: Effetto del rumore ----
noise_values = np.linspace(0, 0.3, 7)
qbers_noise = [simulate_bb84_graph(n_bits=200, noise_prob=n) for n in noise_values]

plt.figure(figsize=(8,4))
plt.plot(noise_values, qbers_noise, marker='o', linewidth=2)
plt.title("Effetto del rumore sul QBER")
plt.xlabel("Probabilità di rumore")
plt.ylabel("QBER (%)")
plt.grid(True)
plt.show()

# ---- Grafico 2: QBER con e senza intercettazione ----
no_eve = simulate_bb84_graph(n_bits=100, noise_prob=0.0, eve_prob=0.0)
with_eve = simulate_bb84_graph(n_bits=100, noise_prob=0.0, eve_prob=0.3)

plt.figure(figsize=(6,4))
plt.bar(["No Eve", "With Eve"], [no_eve, with_eve], color=['green','red'])
plt.title("QBER With vs Without Unauthorized Interception")
plt.ylabel("QBER (%)")
plt.ylim(0,5)
plt.show()

# ==========================
# Simulazione Aer reale
# ==========================

sim = AerSimulator()
transpiled_circuit = transpile(qc, sim)
result = sim.run(transpiled_circuit, shots=1024).result()
counts = result.get_counts()

print("Counts dall'esecuzione Aer:", counts)

# ==========================
# QBER vs lunghezza chiave
# ==========================

def qber_vs_key_length(lengths, noise_level=0.0, intercept=False, shots=20, readout_noise=True):
    avg_qbers = []
    for n in lengths:
        qber_samples = []
        for _ in range(shots):
            qber, *_ = BB84_single_circuit(
                n=n,
                noise_level=noise_level,
                intercept=intercept,
                readout_noise=readout_noise
            )
            qber_samples.append(qber * 100)
        avg_qbers.append(np.mean(qber_samples))
    return avg_qbers

key_lengths = [5, 10, 20, 50, 100, 200]

qbers_no_noise  = qber_vs_key_length(key_lengths, noise_level=0, intercept=False, readout_noise=True)
qbers_noise     = qber_vs_key_length(key_lengths, noise_level=0.05, intercept=False, readout_noise=True)
qbers_eve       = qber_vs_key_length(key_lengths, noise_level=0, intercept=True, readout_noise=True)

plt.figure(figsize=(8,5))
plt.plot(key_lengths, qbers_no_noise, marker='o', label="No Noise, No Eve")
plt.plot(key_lengths, qbers_noise, marker='s', label="Noise 5%")
plt.plot(key_lengths, qbers_eve, marker='^', label="Eve Intercepts")
plt.xlabel("Lunghezza chiave (qubit)")
plt.ylabel("QBER medio (%)")
plt.title("Effetto della lunghezza della chiave sulla sicurezza BB84")
plt.grid(True)
plt.legend()
plt.show()