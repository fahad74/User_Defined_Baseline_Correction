#!/usr/bin/env python3
"""
Baseline Correction Tool (FTIR-ready) with AI Integration
=========================================================
Requirements:
    numpy, pandas, matplotlib, openpyxl, scikit-learn
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd
import math

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

class BaselineCorrectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Baseline Correction & AI Training Tool")
        self.geometry("1280x850")
        self.minsize(950, 700)

        # ---- Data holders ----
        self.x_data = None          
        self.y_data = None          
        self.loaded_filename = None
        self.baseline_points = []   
        self.corrected_y = None     
        self.baseline_curve = None  
        
        # ---- AI Training holders ----
        self.training_data_X = []
        self.training_data_y = []
        self.ai_model = None
        self.ai_baseline_curve = None
        self.ai_corrected_y = None

        # UI state variables
        self.x_reversed = tk.BooleanVar(value=False)
        self.is_absorbance = tk.BooleanVar(value=False)
        self.picking_enabled = tk.BooleanVar(value=False)

        self._build_ui()

    def get_current_y(self):
        if self.y_data is None:
            return None
        if self.is_absorbance.get():
            return 100.0 - self.y_data
        return self.y_data

    def _build_ui(self):
        # ---------------- Top toolbar ----------------
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Button(toolbar, text="Browse Data...", command=self.load_data).pack(side=tk.LEFT, padx=3)

        ttk.Checkbutton(
            toolbar, text="Reverse X Axis (FTIR)",
            variable=self.x_reversed, command=self.update_plots
        ).pack(side=tk.LEFT, padx=8)

        ttk.Checkbutton(
            toolbar, text="Absorbance Mode (100 - Y)",
            variable=self.is_absorbance, command=self.on_absorbance_toggle
        ).pack(side=tk.LEFT, padx=8)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(toolbar, text="Export Data...", command=self.export_data).pack(side=tk.LEFT, padx=3)

        # ---------------- Tabs ----------------
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tab_manual = ttk.Frame(self.notebook)
        self.tab_ai = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_manual, text="Manual Baseline & Training Gen")
        self.notebook.add(self.tab_ai, text="AI Auto-Correction & Comparison")

        self._build_manual_tab()
        self._build_ai_tab()

        # ---------------- Status bar ----------------
        self.status = tk.StringVar(value="Load a data file to begin.")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    def _build_manual_tab(self):
        tools = ttk.Frame(self.tab_manual)
        tools.pack(fill=tk.X, pady=4)
        
        ttk.Label(tools, text="Target # points:").pack(side=tk.LEFT)
        self.npoints_var = tk.IntVar(value=5)
        ttk.Spinbox(tools, from_=2, to=100, textvariable=self.npoints_var, width=5).pack(side=tk.LEFT, padx=3)
        
        self.pick_check = ttk.Checkbutton(
            tools, text="Click on Plot to Add Points",
            variable=self.picking_enabled, command=self._sync_pick_state
        )
        self.pick_check.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(tools, text="Clear Points", command=self.clear_points).pack(side=tk.LEFT, padx=3)
        ttk.Button(tools, text="Subtract Baseline", command=self.subtract_baseline).pack(side=tk.LEFT, padx=3)
        ttk.Separator(tools, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        
        # Add to AI Training button
        ttk.Button(tools, text="Add Current to AI Training Set", command=self.add_to_training).pack(side=tk.LEFT, padx=3)
        self.train_count_lbl = ttk.Label(tools, text="Training samples: 0", foreground="blue")
        self.train_count_lbl.pack(side=tk.LEFT, padx=10)

        body = ttk.Frame(self.tab_manual)
        body.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(body)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_man = Figure(figsize=(8, 6), dpi=100)
        self.ax_man = self.fig_man.add_subplot(111)
        self.canvas_man = FigureCanvasTkAgg(self.fig_man, master=plot_frame)
        self.canvas_man.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.nav_toolbar_man = NavigationToolbar2Tk(self.canvas_man, plot_frame)
        self.nav_toolbar_man.update()
        self.canvas_man.mpl_connect("button_press_event", self.on_click)

        side = ttk.Frame(body, width=320)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        columns = ("x", "y")
        self.tree = ttk.Treeview(side, columns=columns, show="headings", height=18)
        self.tree.heading("x", text="X")
        self.tree.heading("y", text="Y")
        self.tree.column("x", width=140, anchor=tk.CENTER)
        self.tree.column("y", width=140, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=2)
        
        ttk.Button(side, text="Remove Selected Point", command=self.remove_selected_point).pack(fill=tk.X, pady=2)

    def _build_ai_tab(self):
        tools = ttk.Frame(self.tab_ai)
        tools.pack(fill=tk.X, pady=4)
        
        ttk.Button(tools, text="Train AI Model", command=self.train_ai_model).pack(side=tk.LEFT, padx=3)
        ttk.Button(tools, text="Auto-Correct Current Data", command=self.apply_ai_model).pack(side=tk.LEFT, padx=3)
        ttk.Button(tools, text="Compare Accuracy (User vs AI)", command=self.compare_accuracy).pack(side=tk.LEFT, padx=3)
        
        self.ai_metrics_lbl = ttk.Label(tools, text="", foreground="green")
        self.ai_metrics_lbl.pack(side=tk.LEFT, padx=10)

        plot_frame = ttk.Frame(self.tab_ai)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig_ai = Figure(figsize=(8, 6), dpi=100)
        self.ax_ai = self.fig_ai.add_subplot(111)
        self.canvas_ai = FigureCanvasTkAgg(self.fig_ai, master=plot_frame)
        self.canvas_ai.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.nav_toolbar_ai = NavigationToolbar2Tk(self.canvas_ai, plot_frame)
        self.nav_toolbar_ai.update()

    def load_data(self):
        path = filedialog.askopenfilename(title="Select XY data file")
        if not path:
            return
        try:
            # --- UPDATED ROBUST LOADING LOGIC ---
            if path.lower().endswith('.csv'):
                df = pd.read_csv(path, header=None)
            elif path.lower().endswith(('.txt', '.dat')):
                # Explicitly handle any amount of whitespace (spaces or tabs)
                df = pd.read_csv(path, sep=r'\s+', engine="python", header=None)
            else:
                df = pd.read_excel(path, header=None)
            
            # Force take only the first two columns and coerce to numeric
            df = df.iloc[:, :2]
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.empty:
                raise ValueError("No valid numeric data found. Please check the file formatting.")
            
            x = df.iloc[:, 0].to_numpy()
            y = df.iloc[:, 1].to_numpy()
            order = np.argsort(x)
            self.x_data, self.y_data = x[order], y[order]
            self.loaded_filename = os.path.basename(path)

            self.baseline_points, self.corrected_y, self.baseline_curve = [], None, None
            self.ai_baseline_curve, self.ai_corrected_y = None, None
            
            self.refresh_tree()
            self.status.set(f"Loaded '{self.loaded_filename}' — {len(self.x_data)} points.")
            self.update_plots()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def on_absorbance_toggle(self):
        self.baseline_points, self.corrected_y, self.baseline_curve = [], None, None
        self.refresh_tree()
        self.update_plots()

    def update_plots(self):
        self._update_plot_ax(self.ax_man, self.canvas_man, mode="manual")
        self._update_plot_ax(self.ax_ai, self.canvas_ai, mode="ai")

    def _update_plot_ax(self, ax, canvas, mode):
        ax.clear()
        current_y = self.get_current_y()

        if self.x_data is not None and current_y is not None:
            raw_label = "Absorbance" if self.is_absorbance.get() else "Raw Transmission"
            ax.plot(self.x_data, current_y, color="#1f77b4", lw=1.2, label=raw_label)

            if mode == "manual":
                if self.baseline_points:
                    pts = sorted(self.baseline_points, key=lambda p: p[0])
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o--", color="#d62728", ms=6, label="Picked points")
                if self.baseline_curve is not None:
                    ax.plot(self.x_data, self.baseline_curve, color="#ff7f0e", lw=1.5, ls="--", label="Manual Baseline")
                if self.corrected_y is not None:
                    ax.plot(self.x_data, self.corrected_y, color="#2ca02c", lw=1.2, label="Manual Corrected")
                    
            elif mode == "ai":
                # Show AI baseline
                if self.ai_baseline_curve is not None:
                    ax.plot(self.x_data, self.ai_baseline_curve, color="#9467bd", lw=1.5, ls="-.", label="AI Auto Baseline")
                if self.ai_corrected_y is not None:
                    ax.plot(self.x_data, self.ai_corrected_y, color="#e377c2", lw=1.2, label="AI Corrected")
                # Show manual for comparison if it exists
                if self.baseline_curve is not None:
                    ax.plot(self.x_data, self.baseline_curve, color="#ff7f0e", lw=1, ls="--", alpha=0.6, label="Manual Baseline (Reference)")

            ax.legend(loc="best", fontsize=8)
            ax.set_title(self.loaded_filename)
            ax.set_xlabel("Wavenumber (cm⁻¹)" if self.x_reversed.get() else "X")
            ax.set_ylabel("Absorbance" if self.is_absorbance.get() else "Transmission")
            
            if self.x_reversed.get():
                ax.set_xlim(np.max(self.x_data), np.min(self.x_data))
            else:
                ax.set_xlim(np.min(self.x_data), np.max(self.x_data))
        else:
            ax.set_title("Load a data file to begin")

        canvas.figure.tight_layout()
        canvas.draw()

    def _sync_pick_state(self):
        if self.picking_enabled.get():
            self.status.set("Click on the plot to add baseline points.")
        else:
            self.status.set("Point picking disabled.")

    def on_click(self, event):
        if not self.picking_enabled.get() or event.inaxes != self.ax_man or self.nav_toolbar_man.mode != "":
            return
        if event.xdata is None or event.ydata is None:
            return

        self.baseline_points.append([float(event.xdata), float(event.ydata)])
        self.refresh_tree()
        self.update_plots()

        if len(self.baseline_points) >= self.npoints_var.get():
            self.picking_enabled.set(False)

    def refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        self.baseline_points = sorted(self.baseline_points, key=lambda p: p[0])
        for i, (x, y) in enumerate(self.baseline_points):
            self.tree.insert("", tk.END, iid=str(i), values=(f"{x:.6g}", f"{y:.6g}"))

    def remove_selected_point(self):
        sel = self.tree.selection()
        if sel:
            for idx in sorted((int(i) for i in sel), reverse=True):
                del self.baseline_points[idx]
            self.refresh_tree()
            self.update_plots()

    def clear_points(self):
        self.baseline_points, self.corrected_y, self.baseline_curve = [], None, None
        self.refresh_tree()
        self.update_plots()

    def subtract_baseline(self):
        if self.x_data is None or len(self.baseline_points) < 2:
            return
        bx = np.array([p[0] for p in self.baseline_points])
        by = np.array([p[1] for p in self.baseline_points])
        self.baseline_curve = np.interp(self.x_data, bx, by, left=by[0], right=by[-1])
        self.corrected_y = self.get_current_y() - self.baseline_curve
        self.update_plots()

    # ------------------------------------------------------------------
    # AI Training & Application
    # ------------------------------------------------------------------
    def extract_features(self, x, y):
        # Creates features for the ML model: x, y, and local rolling means
        df = pd.DataFrame({'x': x, 'y': y})
        df['y_roll_10'] = df['y'].rolling(10, center=True, min_periods=1).mean()
        df['y_roll_50'] = df['y'].rolling(50, center=True, min_periods=1).mean()
        return df.values

    def add_to_training(self):
        if self.baseline_curve is None:
            messagebox.showwarning("Error", "Subtract manual baseline first to generate the curve.")
            return
        
        current_y = self.get_current_y()
        features = self.extract_features(self.x_data, current_y)
        
        self.training_data_X.append(features)
        self.training_data_y.append(self.baseline_curve)
        
        self.train_count_lbl.config(text=f"Training samples: {len(self.training_data_X)}")
        self.status.set(f"Added current spectra to training set. Total: {len(self.training_data_X)}")

    def train_ai_model(self):
        if len(self.training_data_X) < 1:
            messagebox.showerror("Error", "Add at least 1 dataset (ideally ~10) to training set first.")
            return
        
        self.status.set("Training AI model... Please wait.")
        self.update()
        
        X_train = np.vstack(self.training_data_X)
        y_train = np.concatenate(self.training_data_y)
        
        # Random Forest is highly robust for non-linear localized signal fitting
        self.ai_model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
        self.ai_model.fit(X_train, y_train)
        
        self.status.set("AI model trained successfully!")
        messagebox.showinfo("Success", "Model training complete. You can now auto-correct baselines.")

    def apply_ai_model(self):
        if self.ai_model is None:
            messagebox.showerror("Error", "Train the AI model first.")
            return
        if self.x_data is None:
            return
            
        current_y = self.get_current_y()
        features = self.extract_features(self.x_data, current_y)
        
        self.ai_baseline_curve = self.ai_model.predict(features)
        self.ai_corrected_y = current_y - self.ai_baseline_curve
        
        self.status.set("AI baseline applied.")
        self.notebook.select(self.tab_ai)
        self.update_plots()

    def compare_accuracy(self):
        if self.baseline_curve is None or self.ai_baseline_curve is None:
            messagebox.showerror("Error", "Ensure both Manual and AI baselines are generated for the current file.")
            return
        
        rmse = math.sqrt(mean_squared_error(self.baseline_curve, self.ai_baseline_curve))
        mae = mean_absolute_error(self.baseline_curve, self.ai_baseline_curve)
        
        metrics_text = f"Comparison Metrics | RMSE: {rmse:.4f} | MAE: {mae:.4f}"
        self.ai_metrics_lbl.config(text=metrics_text)
        self.status.set(metrics_text)
        messagebox.showinfo("Accuracy Metrics", f"Root Mean Squared Error (RMSE): {rmse:.4f}\nMean Absolute Error (MAE): {mae:.4f}\n\nLower values indicate the AI baseline closely matches your manual baseline.")

    def export_data(self):
        if self.x_data is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")])
        if not path:
            return

        data = {"X": self.x_data, "Y_raw": self.y_data}
        if self.is_absorbance.get(): data["Y_absorbance"] = 100.0 - self.y_data
        if self.baseline_curve is not None: data["Manual_Baseline"] = self.baseline_curve
        if self.corrected_y is not None: data["Manual_Corrected"] = self.corrected_y
        if self.ai_baseline_curve is not None: data["AI_Baseline"] = self.ai_baseline_curve
        if self.ai_corrected_y is not None: data["AI_Corrected"] = self.ai_corrected_y

        df = pd.DataFrame(data)
        if self.x_reversed.get(): df = df.iloc[::-1].reset_index(drop=True)

        if path.endswith(".xlsx"): df.to_excel(path, index=False)
        else: df.to_csv(path, index=False)
        self.status.set("Export complete.")

if __name__ == "__main__":
    BaselineCorrectionApp().mainloop()