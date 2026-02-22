# BB84 Quantum Key Distribution Simulation

This repository contains a simulation of the BB84 Quantum Key Distribution (QKD) protocol, originally developed during an IBM Qiskit Hackathon. The project analyzes the impact of quantum noise and unauthorized eavesdropping on the security and integrity of the generated cryptographic key.

## Key Features

* **BB84 Circuit Implementation:** Complete simulation of Alice's state preparation, Bob's measurement, and the subsequent sifting phase to extract the shared key.
* **Realistic Noise Modeling:** Integration of custom noise models using `qiskit_aer.noise`, featuring configurable depolarizing errors and readout errors.
* **Eavesdropping Simulation:** Implementation of an intercept-resend attack by a third party (Eve) to observe its measurable impact on the Quantum Bit Error Rate (QBER).
* **Statistical Visualization:** Automated generation of `matplotlib` plots to analyze:
  1. The effect of varying noise probabilities on the QBER.
  2. The QBER discrepancy between secure channels and intercepted channels.
  3. QBER trends across different quantum key lengths.

## Installation

Ensure Python is installed on your system. Install the required dependencies using pip:

    pip install qiskit qiskit-aer matplotlib numpy

## Usage

Clone the repository and execute the main script. 

    git clone <YOUR_REPOSITORY_URL>
    cd <REPOSITORY_NAME>
    python main.py

Upon execution, the script will output the results of a single protocol run to the console, detailing the generated bits, chosen bases, and the resulting QBER. Following the console output, the script will sequentially render three analytical graphs.

## Code Structure

* `get_noise_model()`: Constructs a configurable Qiskit Aer noise model.
* `BB84_single_circuit()`: The core function handling the quantum circuit creation, applying logic gates based on basis selection, simulating Eve's interference, and executing measurements.
* `simulate_bb84_graph()`: A lightweight probabilistic simulation designed to rapidly generate large datasets required for noise and eavesdropping visualizations.
* `qber_vs_key_length()`: A statistical aggregation function that runs multiple Qiskit circuits of varying sizes to compute average QBER metrics.

## Acknowledgments

Developed as part of the IBM Qiskit Hackathon.
