import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from ttkbootstrap.widgets.scrolled import ScrolledText 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import threading
from PIL import Image, ImageTk
from tkinter import filedialog
import shutil 
import os
import simo_backend 

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')

class SimoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 6: SIMO Receiver Simulator (Master Edition)")
        self.root.geometry("1200x850")
        
        style = ttk.Style()
        NAVY_BLUE = "#003366"
        HOVER_BLUE = "#4A90E2"
        TEXT_WHITE = "#FFFFFF"
        
        style.configure('TNotebook.Tab', font=('Segoe UI', 11, 'bold'), padding=[25, 12], borderwidth=0)
        style.map("TNotebook.Tab",
            background=[("selected", NAVY_BLUE), ("active", HOVER_BLUE)],
            foreground=[("selected", TEXT_WHITE), ("active", TEXT_WHITE)],
            expand=[("selected", [1, 1, 1, 0])]
        )
        style.configure('TLabelframe', font=('Segoe UI', 10, 'bold'))
        style.configure('TLabelframe.Label', foreground=NAVY_BLUE)

        # --- Header ---
        header_frame = ttk.Frame(root, bootstyle="primary")
        header_frame.pack(side=TOP, fill=X)
        
        lbl_title = ttk.Label(header_frame, text="Wireless Project 6 | SIMO Receiver Simulator", 
                              font=("Segoe UI", 18, "bold"), bootstyle="inverse-primary", padding=15)
        lbl_title.pack(side=LEFT)
        
        lbl_info = ttk.Label(header_frame, text="SC & MRC Diversity Analysis", 
                                 font=("Segoe UI", 12), bootstyle="inverse-primary", padding=15)
        lbl_info.pack(side=RIGHT)

        # --- Main Layout ---
        main_container = ttk.Frame(root)
        main_container.pack(fill=BOTH, expand=True, padx=15, pady=15)

        # === Left Sidebar ===
        sidebar = ttk.Labelframe(main_container, text=" System Configuration ", bootstyle="default", padding=15)
        sidebar.pack(side=LEFT, fill=Y, padx=(0, 15))
        
        # Input 1: Antenna Count
        ttk.Label(sidebar, text="Number of Antennas (N):", font=("Segoe UI", 10, "bold"), foreground="#555").pack(anchor=W, pady=(10, 5))
        
        self.n_var = ttk.StringVar(value="2")
        self.n_combo = ttk.Combobox(sidebar, textvariable=self.n_var, 
                                    values=["2", "3", "4", "5", "6", "7", "8"], 
                                    state="readonly", bootstyle="primary", font=("Segoe UI", 11))
        self.n_combo.pack(fill=X, pady=5)
        self.n_combo.current(0)

        # Input 2: SNR
        ttk.Label(sidebar, text="Average SNR (dB):", font=("Segoe UI", 10, "bold"), foreground="#555").pack(anchor=W, pady=(10, 5))
        self.snr_var = ttk.StringVar(value="10.0") 
        self.snr_entry = ttk.Entry(sidebar, textvariable=self.snr_var, bootstyle="primary")
        self.snr_entry.pack(fill=X, pady=5)
        
        # Input 3: Fading Model
        ttk.Label(sidebar, text="Fading Model:", font=("Segoe UI", 10, "bold"), foreground="#555").pack(anchor=W, pady=(10, 5))
        self.fading_var = ttk.StringVar(value="Rayleigh")
        self.fading_combo = ttk.Combobox(sidebar, textvariable=self.fading_var, 
                                         values=["Rayleigh", "Rician"], state="readonly", bootstyle="primary")
        self.fading_combo.pack(fill=X, pady=5)
        self.fading_combo.current(0)

        ttk.Separator(sidebar, bootstyle="secondary").pack(fill=X, pady=25)

        # Buttons
        self.run_btn = ttk.Button(sidebar, text="▶  START SIMULATION", 
                                  bootstyle="primary", width=22, command=self.start_simulation_thread)
        self.run_btn.pack(pady=8)

        self.diag_btn = ttk.Button(sidebar, text="📷  GENERATE DIAGRAMS", 
                                  bootstyle="info-outline", width=22, command=self.generate_and_show_diagrams)
        self.diag_btn.pack(pady=8)
        
        # Progress
        self.progress = ttk.Progressbar(sidebar, bootstyle="success-striped", mode='indeterminate')
        self.progress.pack(fill=X, pady=20)
        
        self.status_var = ttk.StringVar(value="Ready.")
        ttk.Label(sidebar, textvariable=self.status_var, font=("Segoe UI", 9, "italic"), foreground="#888").pack()

        # === Right Content ===
        content_frame = ttk.Frame(main_container)
        content_frame.pack(side=LEFT, fill=BOTH, expand=True)

        self.notebook = ttk.Notebook(content_frame, bootstyle="default")
        self.notebook.pack(fill=BOTH, expand=True)

        self.tab_report = ttk.Frame(self.notebook)
        self.tab_snr_plot = ttk.Frame(self.notebook)
        self.tab_ber_plot = ttk.Frame(self.notebook)
        self.tab_diagrams = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_report, text=" 📄 Report ")
        self.notebook.add(self.tab_snr_plot, text=" 📈 SNR Plot ")
        self.notebook.add(self.tab_ber_plot, text=" 📉 BER Plot ")
        self.notebook.add(self.tab_diagrams, text=" 🧩 Diagrams ")
        
        # -- Report Tab --
        report_controls = ttk.Frame(self.tab_report)
        report_controls.pack(fill=X, padx=10, pady=10)
        ttk.Button(report_controls, text="💾 Export Report", command=self.save_report_to_file, bootstyle="secondary-outline").pack(side=RIGHT)
        
        self.report_text = ScrolledText(self.tab_report, font=("Consolas", 11), padding=15)
        self.report_text.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # -- Diagrams Tab --
        self.canvas_diag = ttk.Canvas(self.tab_diagrams)
        self.scroll_x = ttk.Scrollbar(self.tab_diagrams, orient="horizontal", command=self.canvas_diag.xview)
        self.scroll_y = ttk.Scrollbar(self.tab_diagrams, orient="vertical", command=self.canvas_diag.yview)
        self.scrollable_frame = ttk.Frame(self.canvas_diag)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas_diag.configure(scrollregion=self.canvas_diag.bbox("all")))
        self.canvas_diag.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_diag.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        self.scroll_x.pack(side=BOTTOM, fill=X)
        self.scroll_y.pack(side=RIGHT, fill=Y)
        self.canvas_diag.pack(side=LEFT, fill=BOTH, expand=True)

        diag_ctrl_frame = ttk.Frame(self.scrollable_frame)
        diag_ctrl_frame.pack(side=TOP, fill=X, padx=10, pady=10)
        
        ttk.Button(diag_ctrl_frame, text="💾 Save SC Image", bootstyle="info", 
                   command=lambda: self.export_diagram("sc_diagram.png")).pack(side=LEFT, padx=20)
        
        ttk.Button(diag_ctrl_frame, text="💾 Save MRC Image", bootstyle="success", 
                   command=lambda: self.export_diagram("mrc_diagram.png")).pack(side=RIGHT, padx=20)

        images_container = ttk.Frame(self.scrollable_frame)
        images_container.pack(side=TOP, fill=BOTH, expand=True, padx=5, pady=5)

        self.frame_sc_img = ttk.Labelframe(images_container, text=" Selection Combining (SC) ", bootstyle="info")
        self.frame_sc_img.pack(side=LEFT, fill=BOTH, expand=True, padx=10)
        
        self.frame_mrc_img = ttk.Labelframe(images_container, text=" Maximum Ratio Combining (MRC) ", bootstyle="success")
        self.frame_mrc_img.pack(side=LEFT, fill=BOTH, expand=True, padx=10)
        
        self.lbl_sc = ttk.Label(self.frame_sc_img, text="Click 'Generate Diagrams' to view.")
        self.lbl_sc.pack(expand=True)
        
        self.lbl_mrc = ttk.Label(self.frame_mrc_img, text="Click 'Generate Diagrams' to view.")
        self.lbl_mrc.pack(expand=True)

    # --- Logic ---

    def start_simulation_thread(self):
        # 1. Validation
        raw_snr = self.snr_var.get().strip()
        try:
            snr_val = float(raw_snr)
        except ValueError:
            Messagebox.show_error("Invalid SNR Input!\nPlease enter a valid number.", "Input Error")
            return

        if snr_val < -20 or snr_val > 60:
             Messagebox.show_error("SNR Value out of range!\nPlease enter a realistic SNR value (-20 to 60 dB).", "Input Error")
             return

        try:
            n_val = int(self.n_var.get())
            if n_val < 2 or n_val > 8: raise ValueError
        except ValueError:
             Messagebox.show_error("Invalid Antenna Count!", "Input Error")
             return

        self.run_btn.config(state="disabled")
        self.n_combo.config(state="disabled") 
        
        fading_val = self.fading_var.get()
        self.progress.start(10)
        self.status_var.set(f"Simulating for N={n_val}, SNR={snr_val}dB...")

        threading.Thread(target=self.run_simulation, args=(n_val, snr_val, fading_val), daemon=True).start()

    def run_simulation(self, N, snr_db, fading):
        try:
            snr_linear = 10**(snr_db / 10)

            # Backend Call
            results = simo_backend.simo_simulation(N, snr_linear, fading, num_trials=5000)
            avg_snr_sc, avg_snr_mrc, avg_ber_sc, avg_ber_mrc, _, _, _, _ = results
            
            # Report Update
            report_content = f"""
            SIMULATION REPORT: PROJECT 6
            ============================
            
            [1] SYSTEM CONFIGURATION
            ------------------------
            • Antennas (N):       {N}
            • Input SNR:          {snr_db} dB
            • Fading Channel:     {fading}
            
            [2] PERFORMANCE RESULTS (At SNR = {snr_db} dB)
            ------------------------
            A. Output SNR:
               - SC Output:       {10*np.log10(avg_snr_sc):.2f} dB
               - MRC Output:      {10*np.log10(avg_snr_mrc):.2f} dB
               >> Gain (MRC-SC):  +{10*np.log10(avg_snr_mrc) - 10*np.log10(avg_snr_sc):.2f} dB
            
            B. Bit Error Rate (BER):
               - SC BER:          {avg_ber_sc:.6f}
               - MRC BER:         {avg_ber_mrc:.6f}
            
            [3] CONCLUSION
            ------------------------
            MRC provides superior diversity gain compared to SC.
            """
            self.root.after(0, lambda: self._update_report(report_content))

            # --- PLOT 1: SNR Improvement ---
            n_range = list(range(1, 9))
            gain_sc = []; gain_mrc = []
            for n_val in n_range:
                if n_val == 1: 
                    gain_sc.append(0); gain_mrc.append(0)
                else:
                    s_sc, s_mrc, _, _, _, _, _, _ = simo_backend.simo_simulation(n_val, snr_linear, fading, num_trials=2000)
                    gain_sc.append(10*np.log10(s_sc) - snr_db)
                    gain_mrc.append(10*np.log10(s_mrc) - snr_db)

            self.root.after(0, lambda: self.plot_snr_improvement(n_range, gain_sc, gain_mrc, N, snr_db))

            # --- PLOT 2: BER vs SNR ---
            max_snr_plot = max(20, snr_db + 5)
            snr_range_plot = np.linspace(0, max_snr_plot, 11)
            
            ber_sc_list = []; ber_mrc_list = []
            for s_db in snr_range_plot:
                s_lin = 10**(s_db/10)
                _, _, b_sc, b_mrc, _, _, _, _ = simo_backend.simo_simulation(N, s_lin, fading, num_trials=2000)
                ber_sc_list.append(b_sc)
                ber_mrc_list.append(b_mrc)
            
            self.root.after(0, lambda: self.plot_ber_performance(snr_range_plot, ber_sc_list, ber_mrc_list, N, snr_db, avg_ber_sc, avg_ber_mrc))
            
            self.root.after(0, self._simulation_finished)

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self._show_error_safe(error_msg))
            self.root.after(0, self._simulation_finished)

    def _simulation_finished(self):
        self.progress.stop()
        self.status_var.set("Simulation Completed.")
        self.run_btn.config(state="normal")
        self.n_combo.config(state="readonly") 

    def _update_report(self, text):
        self.report_text.delete(1.0, END)
        self.report_text.insert(END, text)

    def _show_error_safe(self, message):
        Messagebox.show_error(message, "Error")

    def save_report_to_file(self):
        content = self.report_text.get("1.0", END)
        if len(content.strip()) == 0:
            Messagebox.show_warning("Report is empty.", "Warning")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")], initialfile="SIMO_Report.txt")
        if file_path:
            with open(file_path, "w") as f:
                f.write(content)
            Messagebox.show_info("Report saved!", "Success")

    def generate_and_show_diagrams(self):
        try:
            self.status_var.set("Generating diagrams...")
            simo_backend.draw_sc_diagram()
            simo_backend.draw_mrc_diagram()
            
            img_sc = Image.open("sc_diagram.png")
            img_mrc = Image.open("mrc_diagram.png")
            
            basewidth = 750 
            wpercent = (basewidth / float(img_sc.size[0]))
            hsize = int((float(img_sc.size[1]) * float(wpercent)))
            img_sc = img_sc.resize((basewidth, hsize), Image.Resampling.LANCZOS)
            
            wpercent_mrc = (basewidth / float(img_mrc.size[0]))
            hsize_mrc = int((float(img_mrc.size[1]) * float(wpercent_mrc)))
            img_mrc = img_mrc.resize((basewidth, hsize_mrc), Image.Resampling.LANCZOS)
            
            self.photo_sc = ImageTk.PhotoImage(img_sc)
            self.photo_mrc = ImageTk.PhotoImage(img_mrc)
            
            self.lbl_sc.configure(image=self.photo_sc, text="")
            self.lbl_mrc.configure(image=self.photo_mrc, text="")
            
            self.notebook.select(self.tab_diagrams)
            self.status_var.set("Diagrams Displayed.")
        except Exception as e:
            Messagebox.show_error(f"Error: {str(e)}\nEnsure 'pillow' is installed.", "Error")

    def export_diagram(self, filename):
        if not os.path.exists(filename):
            Messagebox.show_warning(f"File {filename} not found. Please generate diagrams first.", "Warning")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=filename,
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")], title=f"Save {filename}")
        if file_path:
            try:
                shutil.copy(filename, file_path)
                Messagebox.show_info(f"Saved to: {file_path}", "Success")
            except Exception as e:
                Messagebox.show_error(f"Error saving file: {str(e)}", "Error")

    def save_figure_to_image(self, fig, default_name):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=default_name,
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")], title="Export Plot Image")
        if file_path:
            try:
                fig.savefig(file_path, dpi=300)
                Messagebox.show_info("Saved successfully!", "Success")
            except Exception as e:
                Messagebox.show_error(f"Failed to save: {str(e)}", "Error")

    # --- PROFESSIONAL PLOTTING FUNCTIONS ---

    def _apply_professional_style(self, ax, title, x_label, y_label):
        """Helper to apply publication-quality styling to axes"""
        TEXT_COLOR = '#333333'
        GRID_COLOR = '#E6E6E6'
        
        ax.set_title(title, fontsize=12, fontweight='bold', color='#1A1A1A', pad=15)
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold', color=TEXT_COLOR)
        ax.set_ylabel(y_label, fontsize=10, fontweight='bold', color=TEXT_COLOR)
        
        ax.grid(True, linestyle='--', alpha=0.6, color=GRID_COLOR, zorder=0)
        
        # Despine
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#888888')
        ax.spines['bottom'].set_color('#888888')
        
        ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=9)

    def plot_snr_improvement(self, x, y1, y2, current_n, current_snr):
        for w in self.tab_snr_plot.winfo_children(): w.destroy()
        
        ctrl_frame = ttk.Frame(self.tab_snr_plot)
        ctrl_frame.pack(side=TOP, fill=X, padx=10, pady=5)
        
        # Setup Figure
        fig = Figure(figsize=(5, 4), dpi=100, facecolor='#F8F9FA')
        fig.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.88) 
        ax = fig.add_subplot(111)
        ax.set_facecolor('#FFFFFF')
        
        COLOR_SC = '#2E86C1'
        COLOR_MRC = '#E74C3C'
        
        # Plots
        ax.plot(x, y1, marker='o', linestyle='-', linewidth=2, color=COLOR_SC, markersize=6, label='Selection Combining (SC)')
        ax.plot(x, y2, marker='s', linestyle='-', linewidth=2, color=COLOR_MRC, markersize=6, label='Max Ratio Combining (MRC)')
        
        # Fill Between
        ax.fill_between(x, y1, y2, color=COLOR_MRC, alpha=0.1, label='Diversity Gain Gap')
        
        # Highlight Selection
        if 1 <= current_n <= 8:
            idx = current_n - 1
            ax.scatter([current_n], [y1[idx]], s=150, facecolors='none', edgecolors=COLOR_SC, linewidth=2, zorder=10)
            ax.scatter([current_n], [y2[idx]], s=150, facecolors='none', edgecolors=COLOR_MRC, linewidth=2, zorder=10)
            
            # Annotation
            gain_val = y2[idx] - y1[idx]
            ax.annotate(f'Gain: +{gain_val:.2f} dB',
                        xy=(current_n, (y1[idx] + y2[idx])/2), 
                        xytext=(current_n + 0.5, (y1[idx] + y2[idx])/2),
                        arrowprops=dict(facecolor='#333', arrowstyle='->', alpha=0.7),
                        fontsize=9, color='#333', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd", alpha=0.9))

        self._apply_professional_style(ax, 
                                     f"SNR Improvement Analysis (Input SNR = {current_snr} dB)", 
                                     "Number of Antennas (N)", 
                                     "Output SNR (dB)")
        
        ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9, edgecolor='#ccc')

        ttk.Button(ctrl_frame, text="💾 Export Graph", bootstyle="primary", 
                   command=lambda: self.save_figure_to_image(fig, "SNR_Plot.png")).pack(side=RIGHT)

        canvas = FigureCanvasTkAgg(fig, master=self.tab_snr_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=5, pady=5)
        toolbar = NavigationToolbar2Tk(canvas, self.tab_snr_plot); toolbar.update()

    def plot_ber_performance(self, x, y1, y2, N, current_snr, user_ber_sc, user_ber_mrc):
        for w in self.tab_ber_plot.winfo_children(): w.destroy()
        
        ctrl_frame = ttk.Frame(self.tab_ber_plot)
        ctrl_frame.pack(side=TOP, fill=X, padx=10, pady=5)

        fig = Figure(figsize=(5, 4), dpi=100, facecolor='#F8F9FA')
        fig.subplots_adjust(left=0.14, bottom=0.15, right=0.95, top=0.88)
        ax = fig.add_subplot(111)
        ax.set_facecolor('#FFFFFF')
        
        snr_lin_range = [10**(v/10) for v in x]
        y_theoretical = []
        for s in snr_lin_range:
            term = np.sqrt(s/(1+s))
            y_theoretical.append(0.5*(1-term))
            
        COLOR_THEORY = '#7F8C8D'
        COLOR_SC = '#2980B9'
        COLOR_MRC = '#C0392B'
        
        ax.semilogy(x, y_theoretical, '--', color=COLOR_THEORY, label='Theoretical (SISO)', linewidth=1.5, alpha=0.7)
        ax.semilogy(x, y1, 'o-', color=COLOR_SC, label=f'SC (N={N})', linewidth=2, markersize=5)
        ax.semilogy(x, y2, 's-', color=COLOR_MRC, label=f'MRC (N={N})', linewidth=2, markersize=5)
        
        display_ber_sc = user_ber_sc if user_ber_sc > 0 else 1e-8
        display_ber_mrc = user_ber_mrc if user_ber_mrc > 0 else 1e-8
        
        if current_snr >= 0:
            ax.plot(current_snr, display_ber_sc, 'o', markersize=14, markerfacecolor='none', markeredgecolor=COLOR_SC, markeredgewidth=2)
            ax.plot(current_snr, display_ber_mrc, 's', markersize=14, markerfacecolor='none', markeredgecolor=COLOR_MRC, markeredgewidth=2)
            
            # Text Box with Results
            textstr = '\n'.join((
                r'$\mathbf{Performance \ at \ %.1f \ dB:}$' % (current_snr, ),
                r'$BER_{SC}=%.2e$' % (user_ber_sc, ),
                r'$BER_{MRC}=%.2e$' % (user_ber_mrc, )))
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='bottom', bbox=props)

        self._apply_professional_style(ax, 
                                     f"BER vs SNR Analysis (N={N})", 
                                     "Signal-to-Noise Ratio (dB)", 
                                     "Bit Error Rate (Log Scale)")
        
        ax.set_ylim(bottom=1e-6, top=1)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

        ttk.Button(ctrl_frame, text="💾 Export Graph", bootstyle="primary", 
                   command=lambda: self.save_figure_to_image(fig, "BER_Plot.png")).pack(side=RIGHT)

        canvas = FigureCanvasTkAgg(fig, master=self.tab_ber_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=5, pady=5)
        toolbar = NavigationToolbar2Tk(canvas, self.tab_ber_plot); toolbar.update()

if __name__ == "__main__":
    app_root = ttk.Window(themename="cosmo")
    app = SimoApp(app_root)
    app_root.mainloop()