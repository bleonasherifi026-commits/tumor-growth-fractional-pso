# tumor-growth-fractional-pso
Python implementation of the fractional-order Gompertz model optimized via Particle Swarm Optimization for clinical tumor growth analysis.
import math
import os
import matplotlib
matplotlib.use('Agg')  # Prevents GUI popup windows during image saving

# Font configuration to prevent missing font errors
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['font.enable_last_resort'] = False

import matplotlib.pyplot as plt
import numpy as np


def gamma_func(x):
    """Computes the standard Gamma function."""
    return math.gamma(x)


def solve_fractional_gompertz(V0, r, K, alpha, target_months, dt=0.05):
    """
    Solves the Caputo Fractional Gompertz model using the L1 scheme.
    Supports both tumor growth (K > V) and decay/shrinkage (K < V).
    """
    max_month = float(np.max(target_months))
    num_steps = int(np.ceil(max_month / dt)) + 1
    t_grid = np.linspace(0, max_month, num_steps)
    
    V = np.zeros(num_steps)
    V[0] = V0
    
    coeff = (dt**alpha) * gamma_func(2.0 - alpha)
    r_scaled = r**alpha
    
    for k in range(1, num_steps):
        V_prev = V[k-1]
        if V_prev <= 0:
            V_prev = 1e-5
            
        # Unconstrained logarithmic term allowing both growth (K > V_prev) and decay (K < V_prev)
        growth_term = r_scaled * V_prev * math.log(K / V_prev)
        
        # Memory trace accumulation for fractional order differential equation
        memory_sum = 0.0
        for j in range(k - 1):
            b_jk = ((k - j)**(1.0 - alpha)) - ((k - j - 1)**(1.0 - alpha))
            delta_V = V[j+1] - V[j]
            memory_sum += b_jk * delta_V
            
        V[k] = V_prev + coeff * growth_term - memory_sum
        if V[k] < 1e-5:
            V[k] = 1e-5
            
    V_at_target = np.interp(target_months, t_grid, V)
    return V_at_target, t_grid, V


def compute_fitness(params, V0, t_exp, V_exp):
    """Calculates Mean Squared Error (MSE) between simulated and observed MRI volumes."""
    r, K, alpha = params
    V_num_target, _, _ = solve_fractional_gompertz(V0, r, K, alpha, t_exp)
    mse = np.mean((V_exp - V_num_target)**2)
    return mse


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization (PSO) for parameter identification (r, K, alpha)."""
    def __init__(self, num_particles, iterations, bounds, V0, t_exp, V_exp):
        self.num_particles = num_particles
        self.iterations = iterations
        self.bounds = bounds
        self.V0 = V0
        self.t_exp = t_exp
        self.V_exp = V_exp
        
        self.positions = np.zeros((num_particles, 3))
        self.velocities = np.zeros((num_particles, 3))
        self.pbest_pos = np.zeros((num_particles, 3))
        self.pbest_val = np.full(num_particles, fill_value=np.inf)
        self.gbest_pos = np.zeros(3)
        self.gbest_val = np.inf
        
        for i in range(num_particles):
            for d in range(3):
                low, high = self.bounds[d]
                self.positions[i, d] = np.random.uniform(low, high)
                self.velocities[i, d] = np.random.uniform(-(high-low)*0.1, (high-low)*0.1)
            self.pbest_pos[i] = np.copy(self.positions[i])
            
    def optimize(self):
        w, c1, c2 = 0.5, 1.5, 1.5
        for it in range(self.iterations):
            for i in range(self.num_particles):
                current_fitness = compute_fitness(self.positions[i], self.V0, self.t_exp, self.V_exp)
                
                if current_fitness < self.pbest_val[i]:
                    self.pbest_val[i] = current_fitness
                    self.pbest_pos[i] = np.copy(self.positions[i])
                    
                if current_fitness < self.gbest_val:
                    self.gbest_val = current_fitness
                    self.gbest_pos = np.copy(self.positions[i])
                    
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive = c1 * r1 * (self.pbest_pos[i] - self.positions[i])
                social = c2 * r2 * (self.gbest_pos - self.positions[i])
                self.velocities[i] = w * self.velocities[i] + cognitive + social
                self.positions[i] += self.velocities[i]
                
                # Enforce parameter bounds
                for d in range(3):
                    low, high = self.bounds[d]
                    if self.positions[i, d] < low:
                        self.positions[i, d] = low
                        self.velocities[i, d] *= -0.5
                    elif self.positions[i, d] > high:
                        self.positions[i, d] = high
                        self.velocities[i, d] *= -0.5
                        
        return self.gbest_pos, self.gbest_val


def save_patient_plot(patient_id, V0, estimated_params, t_exp, V_exp, output_folder):
    """Generates and saves visual calibration plots for fractional vs classical models."""
    t_smooth = np.linspace(0, 6, 121)
    
    r_est, K_est, alpha_est = estimated_params
    
    # Simulate fitted Fractional Gompertz trajectory
    _, _, V_frac_fit = solve_fractional_gompertz(V0, r_est, K_est, alpha_est, t_smooth)
    
    # Simulate Classical Gompertz trajectory (alpha fixed at 1.0)
    _, _, V_class_fit = solve_fractional_gompertz(V0, r_est, K_est, 1.0, t_smooth)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    # Plot curves
    ax.plot(t_smooth, V_class_fit, 'r--', label='Classical Gompertz (alpha=1.0)', linewidth=2)
    ax.plot(t_smooth, V_frac_fit, 'b-', label=f'Fractional Gompertz (alpha={alpha_est:.2f})', linewidth=2.5)
    ax.scatter(t_exp, V_exp, color='black', s=60, zorder=5, label='MRI Observations (t0, t1, t2)')
    
    # Title and labels
    ax.set_title(f'Patient {patient_id} - Model Calibration', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (Months after baseline MRI t0)', fontsize=10)
    ax.set_ylabel('Tumor Volume (cm³)', fontsize=10)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(['t0 (0m)', '1m', '2m', '3m', 't1 (4m)', '5m', 't2 (6m)'])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')
    
    # Display parameter panel
    param_text = f"Estimated: r = {r_est:.4f}, K = {K_est:.2f}, alpha = {alpha_est:.4f}"
    fig.text(0.15, 0.02, param_text, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
    
    fig.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.22)
    
    file_name = f"patient{patient_id}_mri_fit.png"
    save_path = os.path.join(output_folder, file_name)
    
    plt.savefig(save_path)
    plt.close(fig)
    plt.close('all')


def main():
    print("==================================================================")
    print(" Caputo Fractional Gompertz Engine - Sequential Patient Dataset")
    print("==================================================================\n")
    
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_folder = os.path.join(desktop_path, "patient_plots")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Figures will be saved in directory: {output_folder}\n")
    
    t_exp = np.array([0.0, 4.0, 6.0])
    
    raw_volumes = [
        np.array([76.44, 66.73, 61.75]),
        np.array([32.67, 35.66, 37.49]),
        np.array([116.44, 40.31, 35.66]),
        np.array([20.61, 12.58, 7.70]),
        np.array([3.76, 9.63, 14.65]),
        np.array([77.22, 60.23, 43.76]),
        np.array([8.75, 4.80, 2.44]),
        np.array([19.87, 4.90, 24.42]),
        np.array([65.26, 80.97, 39.35]),
        np.array([53.31, 68.69, 69.97]),
        np.array([14.50, 15.62, 10.09]),
        np.array([13.65, 81.08, 170.86]),
        np.array([17.04, 11.65, 18.57]),
        np.array([14.51, 50.35, 90.21]),
        np.array([97.24, 41.60, 12.98]),
        np.array([34.75, 34.46, 31.16]),
        np.array([28.72, 24.65, 7.63]),
        np.array([71.41, 33.02, 31.56]),
        np.array([9.43, 11.41, 1.34]),
        np.array([64.98, 48.61, 29.82])
    ]
    
    first_20_volumes = raw_volumes[:20]
    patient_dataset = [{"id": idx + 1, "V_exp": vols} for idx, vols in enumerate(first_20_volumes)]
    
    print(f"Prepared {len(patient_dataset)} patients with IDs from 1 to {len(patient_dataset)}.\n")
    
    for p in patient_dataset:
        V_exp = p["V_exp"]
        V0 = V_exp[0]
        
        max_v = np.max(V_exp)
        min_v = np.min(V_exp)
        
        # Parameter bounds:
        # r in [0.001, 1.0]
        # K in [0.1 * min_v, 3.0 * max_v] (allows K < V0 for tumor decay)
        # alpha in [0.1, 0.95] (ensures visible memory dynamics)
        bounds = [(0.001, 1.0), (0.1 * min_v, 3.0 * max_v), (0.3, 0.95)]
        
        # Enhanced PSO settings for global search performance
        pso = ParticleSwarmOptimizer(
            num_particles=40,
            iterations=100,
            bounds=bounds,
            V0=V0,
            t_exp=t_exp,
            V_exp=V_exp
        )
        best_params, min_mse = pso.optimize()
        
        print(f"Patient {p['id']}: V_exp={V_exp} | Optimal [r, K, alpha]={np.round(best_params, 4)} | MSE={min_mse:.4f}")
        save_patient_plot(p["id"], V0, best_params, t_exp, V_exp, output_folder)

    print(f"\nDone! All 20 plots have been successfully saved to: {output_folder}")


if __name__ == "__main__":
    main()
