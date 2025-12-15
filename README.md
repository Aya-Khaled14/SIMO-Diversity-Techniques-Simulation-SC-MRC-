# SIMO Diversity Techniques Simulation (SC & MRC)

## 📡 Project Overview
This project presents a **physical-layer simulation of a SIMO (Single Input Multiple Output) wireless communication system**, focusing on **diversity combining techniques** to combat fading in wireless channels.

The simulation evaluates and compares:
- **Selection Combining (SC)**
- **Maximum Ratio Combining (MRC)**

under different fading environments using **Rayleigh and Rician channel models**.

The project consists of:
- A **backend simulation module** (physical layer logic)
- A **GUI application** (implemented in a separate file) for user interaction and visualization

---

## 🎯 Objectives
- Study the effect of **antenna diversity** on system performance  
- Compare SC and MRC in terms of:
  - Average SNR
  - Bit Error Rate (BER)
  - Outage Probability
- Provide a **clear educational simulation** suitable for communication systems courses and graduation projects

---

## 🧠 System Model
- **Modulation:** BPSK  
- **Channel Models:**
  - Rayleigh Fading
  - Rician Fading (with configurable K-factor)
- **Receiver Techniques:**
  - Selection Combining (SC)
  - Maximum Ratio Combining (MRC)
- **Noise:** Complex AWGN  

---

## 🗂️ Project Structure

├── simo_backend.py # Physical layer simulation logic
├── main_gui.py # GUI application (run this file)
├── sc_diagram.png # SC block diagram (auto-generated)
├── mrc_diagram.png # MRC block diagram (auto-generated)
└── README.md


---

## ⚙️ Backend Features (`simo_backend.py`)
- Channel generation:
  - Rayleigh fading
  - Rician fading
- SIMO transmission and reception model
- Performance metrics:
  - Instantaneous and average SNR
  - Bit Error Rate (BER)
- Outage probability calculation
- Automatic generation of **professional block diagrams** for:
  - Selection Combining (SC)
  - Maximum Ratio Combining (MRC)

---

## 🖥️ GUI Application
The GUI allows users to:
- Select the number of receive antennas
- Choose fading type (Rayleigh / Rician)
- Set SNR values
- Run simulations interactively
- Visualize performance results
