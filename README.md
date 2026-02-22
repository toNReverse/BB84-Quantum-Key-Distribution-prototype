<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Qiskit-6929C4?style=for-the-badge&logo=qiskit&logoColor=white" alt="Qiskit">
  <img src="https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=matplotlib&logoColor=black" alt="Matplotlib">
  <img src="https://img.shields.io/badge/IBM_Quantum-052FAD?style=for-the-badge&logo=ibm&logoColor=white" alt="IBM">
</p>

# BB84 Quantum Key Distribution & QBER Analysis

*A comprehensive simulation of the BB84 protocol developed during an IBM Qiskit Hackathon, focused on analyzing Quantum Bit Error Rate (QBER) under various noise and interception conditions.*



## 🌟 Project Overview

This project implements the **BB84 protocol**, the foundation of Quantum Key Distribution (QKD). Using **Qiskit**, the simulation explores how the security of a quantum-generated key is affected by environmental noise and the presence of an eavesdropper (Eve). 

The core of the project is the analysis of the **QBER (Quantum Bit Error Rate)**, which is the primary metric used in quantum cryptography to detect potential intrusions.

---

## 📊 Experimental Results & Insights

### 1. Effect of Noise on QBER
The simulation models a realistic quantum channel by introducing **depolarizing noise** and **readout errors**. 
> **Insight:** The QBER increases linearly with the noise probability. In a real-world scenario, a QBER consistently above 11% typically indicates that the key is no longer secure.

<p align="center">
  <img width="600" alt="noise_effect" src="https://github.com/user-attachments/assets/d5f21758-148d-45b5-afeb-6ab3517492ec" />
</p>

### 2. Impact of Eavesdropping (Eve)
This chart compares a clean channel with one subject to an **Intercept-Resend attack**. 
> **Insight:** Even without environmental noise, Eve's presence introduces a significant error rate (theoretically 25% for intercepted bits), making her detection inevitable.

<p align="center">
  <img width="600" alt="eve_impact" src="https://github.com/user-attachments/assets/6bad7b38-2c22-4ab7-89e3-d6ae54d1e4d5" />
</p>

### 3. Key Length vs. Security Stability
Testing different key lengths shows how the average QBER stabilizes as the key grows.
> **Insight:** Longer keys provide a more statistically stable QBER, reducing the chance of "false negatives" where an eavesdropper might go unnoticed due to lucky guesses.

<p align="center">
  <img width="600" alt="key_length_analysis" src="https://github.com/user-attachments/assets/b2be3794-8b86-42fb-a837-18fd1758a944" />
</p>

---

## ✨ Technical Features

* **Quantum Error Simulation:** Implements `NoiseModel` with `depolarizing_error` and `ReadoutError`.
* **Intercept-Resend Attack:** Simulation of an eavesdropper measuring and re-sending qubits.
* **Automated Sifting:** Logic to compare Alice's and Bob's bases and extract the final shared key.
* **Aer Simulator Integration:** High-performance simulation using `AerSimulator`.

---

## 🚀 Installation & Usage

### Requirements
```bash
pip install qiskit qiskit-aer matplotlib numpy
```

### Execution
Run the main simulation script:
```bash
python "BB84 Protocol.py"
```

---

## 📄 License

This project is licensed under the **MIT License**.

> Developed for the **IBM Qiskit Hackathon**.
