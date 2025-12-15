# ==========================================
# FILE NAME: simo_backend.py
# DESCRIPTION: SIMO Receiver Logic (Physical Layer Model)
# AUTHOR: Graduation Project Team
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- Global Constants ---
NUM_TRIALS_DEFAULT = 10000 
NUM_BITS_DEFAULT = 1000  

# --- 1. Channel Generation Function ---
def generate_fading_channels(N, fading_type, K=10):
    """
    Generates complex channel coefficients h based on distribution.
    Normalized such that E[|h|^2] = 1.
    """
    if fading_type == 'Rayleigh':
        # Rayleigh: Real & Imag parts are Gaussian(0, sigma)
        # Power = 2 * sigma^2 = 1 => sigma = sqrt(0.5)
        sigma = np.sqrt(0.5)
        h = (np.random.normal(0, sigma, N) + 1j * np.random.normal(0, sigma, N))
    elif fading_type == 'Rician':
        # Rician: LOS component (mu) + NLOS component (sigma)
        # K = Power_LOS / Power_NLOS
        sigma = np.sqrt(1 / (2 * (1 + K)))
        mu = np.sqrt(K / (1 + K))
        h = (np.random.normal(mu, sigma, N) + 1j * np.random.normal(0, sigma, N))
    else:
        # Fallback
        h = np.ones(N, dtype=complex)
    return h

# --- 2. Main Simulation (Physical Layer) ---
def simo_simulation(N, snr_avg_linear, fading_type, num_trials=NUM_TRIALS_DEFAULT):
    """
    Simulates the full Tx-Rx chain:
    1. Transmit x
    2. Apply Channel h
    3. Add Noise n (y = hx + n)
    4. Apply Combining (SC & MRC)
    5. Detect & Count Errors
    """
    snr_sc_list = []
    snr_mrc_list = []
    ber_sc_list = []
    ber_mrc_list = []
    
    # Noise Standard Deviation Calculation
    # For BPSK (Es=1), SNR = 1/N0. Noise Variance (Complex) = N0.
    # Std Dev per dimension = sqrt(N0 / 2)
    noise_std = np.sqrt(1 / (2 * snr_avg_linear))

    for _ in range(num_trials):
        # A. Channel Generation
        h = generate_fading_channels(N, fading_type)
        
        # Record Instantaneous SNRs (for reporting)
        snr_inst = np.abs(h)**2 * snr_avg_linear
        snr_sc_list.append(np.max(snr_inst))
        snr_mrc_list.append(np.sum(snr_inst))
        
        # B. Data Transmission (Physical Model)
        # Generate random bits (0, 1) -> Symbols (-1, +1)
        bits = np.random.randint(0, 2, NUM_BITS_DEFAULT)
        x = 2*bits - 1 
        
        # Generate Noise Matrix [N_antennas x N_bits]
        noise = (np.random.normal(0, noise_std, (N, NUM_BITS_DEFAULT)) + 
                 1j * np.random.normal(0, noise_std, (N, NUM_BITS_DEFAULT)))
        
        # Received Signal Matrix [y = h*x + n]
        # Broadcasting h to match signal shape
        y = h[:, np.newaxis] * x + noise
        
        # --- C. Selection Combining (SC) ---
        # 1. Identify best antenna (Max |h|)
        best_idx = np.argmax(np.abs(h))
        # 2. Select signal from best antenna
        y_sc = y[best_idx, :]
        # 3. Coherent Detection (Equalization): Divide by channel phase
        # y_est = y / h (Zero Forcing for single tap)
        y_sc_est = y_sc / h[best_idx]
        # 4. Decision (Real part > 0)
        bits_est_sc = (np.real(y_sc_est) > 0).astype(int)
        # 5. BER Calculation
        ber_sc_list.append(np.mean(bits != bits_est_sc))
        
        # --- D. Maximum Ratio Combining (MRC) ---
        # 1. Calculate Weights (w = h*)
        w = np.conj(h)
        # 2. Combine: Sum(w * y)
        # Multiply each antenna's signal by its weight and sum them
        y_mrc = np.sum(w[:, np.newaxis] * y, axis=0)
        # 3. Decision (Real part > 0)
        # Note: MRC naturally aligns phases, so Real part contains the signal energy
        bits_est_mrc = (np.real(y_mrc) > 0).astype(int)
        # 4. BER Calculation
        ber_mrc_list.append(np.mean(bits != bits_est_mrc))

    # Return Averages
    return (np.mean(snr_sc_list), np.mean(snr_mrc_list), 
            np.mean(ber_sc_list), np.mean(ber_mrc_list), 
            0, 0, 0, 0) # Zeros are placeholders for variances (optional)

# --- 3. Outage Probability ---
def calculate_outage_probability(N, snr_avg_linear, fading_type, threshold_db=10, num_trials=5000):
    thresh_lin = 10**(threshold_db/10)
    outage_sc = 0
    outage_mrc = 0
    
    for _ in range(num_trials):
        h = generate_fading_channels(N, fading_type)
        snr_inst = np.abs(h)**2 * snr_avg_linear
        
        # SC Outage: Best branch < Threshold
        if np.max(snr_inst) < thresh_lin: outage_sc += 1
        # MRC Outage: Sum of branches < Threshold
        if np.sum(snr_inst) < thresh_lin: outage_mrc += 1
            
    return outage_sc/num_trials, outage_mrc/num_trials

# --- 4. Diagram Generation (Professional & Fixed Layout) ---

def draw_sc_diagram():
    """Draws a clear Selection Combining Block Diagram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set limits to prevent cutting off text
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 1. Transmitter (Tx)
    ax.add_patch(Rectangle((0.5, 3.5), 1.5, 1.5, fill=True, color='#add8e6', ec='black'))
    ax.text(1.25, 4.25, "Tx\n(Source)", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 2. Channel Paths
    ax.arrow(2, 4.5, 3, 2, head_width=0.3, head_length=0.3, fc='k', ec='k', alpha=0.6)
    ax.text(3.5, 6, "h_1 + n_1", ha='center', fontsize=11, color='red')
    
    ax.text(3.5, 4.25, ".\n.\n.", ha='center', fontsize=15)
    
    ax.arrow(2, 4, 3, -2, head_width=0.3, head_length=0.3, fc='k', ec='k', alpha=0.6)
    ax.text(3.5, 2.5, "h_N + n_N", ha='center', fontsize=11, color='red')
    
    # 3. Receiver / Selection Block
    ax.add_patch(Rectangle((5.5, 1.5), 3, 5.5, fill=True, color='#ffe4b5', ec='orange', lw=2))
    ax.text(7, 6.5, "Rx 1 (SNR_1)", ha='center', fontsize=9)
    ax.text(7, 2.0, "Rx N (SNR_N)", ha='center', fontsize=9)
    
    # Logic Block
    ax.add_patch(Rectangle((6, 3.5), 2, 1.5, fill=True, color='orange', ec='black'))
    ax.text(7, 4.25, "Selection\nLogic\nMax(SNR)", ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 4. Output
    ax.arrow(8.5, 4.25, 2, 0, head_width=0.3, head_length=0.3, fc='k', ec='k')
    ax.text(10.5, 4.5, "Output Signal\n(Best SNR)", ha='center', fontsize=11)
    
    plt.title("Selection Combining (SC) System Model", fontsize=14)
    plt.tight_layout()
    plt.savefig('sc_diagram.png', dpi=150)
    plt.close()

def draw_mrc_diagram():
    """Draws a clear Maximum Ratio Combining Block Diagram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set limits
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 1. Transmitter
    ax.add_patch(Rectangle((0.5, 3.5), 1.5, 1.5, fill=True, color='#add8e6', ec='black'))
    ax.text(1.25, 4.25, "Tx\n(Source)", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 2. Channels
    ax.arrow(2, 4.5, 2.5, 2, head_width=0.2, fc='k', ec='k', alpha=0.5)
    ax.text(3, 6, "h_1", color='blue', fontsize=10)
    ax.arrow(2, 4, 2.5, -2, head_width=0.2, fc='k', ec='k', alpha=0.5)
    ax.text(3, 2.5, "h_N", color='blue', fontsize=10)
    
    # 3. Weights
    ax.add_patch(Rectangle((5, 6), 1.5, 1, color='yellow', ec='black'))
    ax.text(5.75, 6.5, "x h_1*", ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.add_patch(Rectangle((5, 1.5), 1.5, 1, color='yellow', ec='black'))
    ax.text(5.75, 2, "x h_N*", ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.text(5.75, 4.25, ".\n.\n.", ha='center', fontsize=15)

    # 4. Summation Block
    ax.text(8, 4.25, "Σ", fontsize=40, ha='center', va='center', color='green')
    ax.add_patch(Rectangle((7.5, 3.5), 1, 1.5, fill=False, ec='green', lw=2))
    
    ax.arrow(6.5, 6.5, 1, -2, head_width=0.15, fc='k') 
    ax.arrow(6.5, 2, 1, 1.5, head_width=0.15, fc='k')  
    
    # 5. Output
    ax.arrow(8.5, 4.25, 2, 0, head_width=0.3, fc='k', ec='k')
    ax.text(10.5, 4.5, "Output\n(Max SNR)", ha='center', fontsize=11)
    
    plt.title("Maximum Ratio Combining (MRC) System Model", fontsize=14)
    plt.tight_layout()
    plt.savefig('mrc_diagram.png', dpi=150)
    plt.close()

# --- Main Block (Disabled Auto-Run) ---
if __name__ == "__main__":
    print("\nThis file is a backend module.")
    print("Please run 'main_gui.py' to start the application.\n")