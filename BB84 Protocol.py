from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Funzioni BB84 originali
# ==========================

def get_noise_model(prob_error):
    noise_model = NoiseModel()
    error = depolarizing_error(prob_error, 1)
    noise_model.add_all_qubit_quantum_error(error, ['h', 'x', 'measure'])
    return noise_model

def random_bits(n):
    return [random.randint(0,1) for _ in range(n)]

def random_bases(n):
    return [random.choice(['X','Z']) for _ in range(n)]

def BB84_single_circuit(n, noise_level=0.0, intercept=False):
    alice_bits  = random_bits(n)
    alice_bases = random_bases(n)
    qc = QuantumCircuit(n, n)

    # Codifica Alice
    for i in range(n):
        if alice_bits[i] == 1:
            qc.x(i)
        if alice_bases[i] == 'X':
            qc.h(i)

    # Intercettazione Eve
    if intercept:
        eve_bases = random_bases(n)
        for i in range(n):
            if eve_bases[i] == 'X':
                qc.h(i)
            qc.measure(i, i)
            qc.reset(i)
            if alice_bits[i] == 1:
                qc.x(i)
            if alice_bases[i] == 'X':
                qc.h(i)

    # Basi Bob
    bob_bases = random_bases(n)
    for i in range(n):
        if bob_bases[i] == 'X':
            qc.h(i)
        qc.measure(i, i)

    # Simulazione
    noise_model = get_noise_model(noise_level) if noise_level > 0 else None
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(qc, shots=1).result()
    measured = list(result.get_counts().keys())[0][::-1]
    bob_bits = list(map(int, measured))

    matching = [i for i in range(n) if alice_bases[i] == bob_bases[i]]
    alice_key = [alice_bits[i] for i in matching]
    bob_key   = [bob_bits[i]  for i in matching]

    if len(alice_key) == 0:
        return 0, [], [], 0, qc

    errors = sum(a != b for a,b in zip(alice_key, bob_key))
    qber = errors / len(alice_key)

    return qber, alice_key, bob_key, matching, qc

# ==========================
# Esempio di esecuzione
# ==========================

qber, alice_key, bob_key, matching, qc = BB84_single_circuit(
    n=20,
    noise_level=0.05,
    intercept=False
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

        # Eve intercetta
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
    if len(matching) == 0:
        return 0
    errors = sum(sender_bits[i] != receiver_bits[i] for i in matching)
    return (errors / len(matching)) * 100

# ---- Grafico 1: Effetto del rumore ----
noise_values = np.linspace(0, 0.3, 7)
qbers_noise = [simulate_bb84_graph(n_bits=200, noise_prob=n) for n in noise_values]

plt.figure(figsize=(8,4))
plt.plot(noise_values, qbers_noise, marker='o', linewidth=2, color='blue')
plt.title("Effetto del rumore sul QBER")
plt.xlabel("Probabilità di rumore")
plt.ylabel("QBER (%)")
plt.grid(True)
plt.show()

# ---- Grafico 2: QBER con e senza intercettazione (semplificato) ----
no_eve = simulate_bb84_graph(n_bits=100, noise_prob=0.0, eve_prob=0.0)
with_eve = simulate_bb84_graph(n_bits=100, noise_prob=0.0, eve_prob=0.3)

plt.figure(figsize=(6,4))
plt.bar(["No Eve", "With Eve"], [no_eve, with_eve], color=['green','red'])
plt.title("QBER With vs Without Unauthorized Interception")
plt.ylabel("QBER (%)")
plt.ylim(0,5)
plt.show()

# ==========================
# Simulazione Aer reale aggiunta
# ==========================

# ==========================
# Simulazione Aer reale aggiunta
# ==========================

sim = AerSimulator()
transpiled_circuit = transpile(qc, sim)
result = sim.run(transpiled_circuit, shots=1024).result()
counts = result.get_counts()

print("Counts dall'esecuzione Aer:", counts)