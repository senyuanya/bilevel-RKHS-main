import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde
from scipy.integrate import trapz
import json
import time
import os

# Import path from add_mypath.py
import add_mypathlevy

# Set random number seed
seed = 0
np.random.seed(seed)

def Explicit_Euler(X0, dt, num_space, num_time, Lt):
    start_time = time.time()
    X_explicit = np.zeros((num_space, num_time + 1))
    X_explicit[:, 0] = X0
    for i in range(num_time):
        # Update: X_{t+1} = X_t + ΔL_t
        X_explicit[:, i+1] = X_explicit[:, i] + Lt[:, i]
    
    X_explicit = X_explicit[:, 1:]
    time_explicit = time.time() - start_time
    return X_explicit, time_explicit

def Implicit_Euler(X0, dt, num_space, num_time, Lt):
    start_time = time.time()
    X_implicit = np.zeros((num_space, num_time + 1))
    X_implicit[:, 0] = X0
    # Implicit Euler method iteration
    for k in range(num_time):
        X_k = X_implicit[:, k]
        dL_k = Lt[:, k]
        # Use fsolve to solve the implicit equation
        implicit_f = lambda X_k1: X_k1 - X_k - dL_k
        X_implicit[:, k+1] = fsolve(implicit_f, X_k)
        print("Implicit iteration step:",k)
    time_implicit = time.time() - start_time
    X_implicit = X_implicit[:, 1:]
    return X_implicit, time_implicit

# Save data
def save_data(dt, T, noise_type, initial_type, Data_val, trajectories_val, time_val, method):
    # Create a save folder
    folder = os.path.join(add_mypathlevy.kdedata_folder, f"PureSDE{method}_results_noise_{noise_type}_initial_type{initial_type}_dt_{dt}_T_{T}")
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Save orginal trajectories data
    np.save(os.path.join(folder, f"Data_{method}.npy"), Data_val)
    
    # Save modified trajectories data
    np.save(os.path.join(folder, f"trajectories_{method}.npy"), trajectories_val)
    
    # Save meta data
    metadata = {
        "dt": dt,
        "num_trajectories": trajectories_val.shape[0],
        "num_timepoints": trajectories_val.shape[1],
        "computation time": time_val,
    }
    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(metadata, f)


# Load data
def load_data(dt, T, noise_type, initial_type, method):

    folder = os.path.join(add_mypathlevy.kdedata_folder, f"PureSDE{method}_results_noise_{noise_type}_initial_type{initial_type}_dt_{dt}_T_{T}")
    
    # Load original trajectories
    Data_set = np.load(os.path.join(folder, f"Data_{method}.npy"))
    
    # Load modified trajectories
    trajectories_set = np.load(os.path.join(folder, f"trajectories_{method}.npy"))
    
    # Load meta data
    with open(os.path.join(folder, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    return Data_set, trajectories_set, metadata


def simulate_SDEPureJump(dt, T, Nx, X0, Lt, noise_type, initial_type, method='Explicit'):
    """
    Simulate the trajectories data with the implicit Euler method or the explicit Euler method.
    """

    data_filename = os.path.join(add_mypathlevy.kdedata_folder, f"PureSDE{method}_results_noise_{noise_type}_initial_type{initial_type}_dt_{dt}_T_{T}")
    
    # Check if data already exists
    if os.path.exists(data_filename):
        print("Trajectory Data file already exists. Loading data...")       
        Dataset, trajectories, metaset = load_data(dt, T, noise_type, initial_type, method)
        timeset = metaset['computation time']
        
    else:
        print("Trajectory Data file does not exist. Generating data...")
        Nt = int(T / dt)
        t = np.linspace(0, T, Nt + 1)
        
        if method == "Explicit":
            Dataset, timeset = Explicit_Euler(X0, dt, Nx, Nt, Lt)
        else:
            Dataset, timeset = Implicit_Euler(X0, dt, Nx, Nt, Lt)
            
        # Check and remove trajectories containing infinity or NaN values, while also handling extreme values 
        exvalid_indices = np.all(np.isfinite(Dataset), axis=1)
        trajectories = Dataset[exvalid_indices,:]
        
        #Save data
        save_data(dt, T, noise_type, initial_type, Dataset, trajectories, timeset, method)

        print("-"* 10 + f" Save {method} SDE data successfully " + "-" * 10)

    return  Dataset, trajectories, timeset
        

def compare_explicit_implicit(dt, T, noise_type, initial_type, Data_explicit, Data_implicit, time_explicit, time_implicit, plot_on = True, figure_format='pdf', dpi=300):
    
        # Check and remove trajectories containing infinity or NaN values, while also handling extreme values 
        exvalid_indices = np.all(np.isfinite(Data_explicit), axis=1)
        trajectories_explicit = Data_explicit[exvalid_indices,:]
        trajectories_implicit = Data_implicit[exvalid_indices,:]
        
        # obtain the final state
        final_states_explicit = trajectories_explicit[:, -1]
        final_states_implicit = trajectories_implicit[:, -1]
        
    #     print("final_states_explicit:",final_states_explicit)
    #     print("final_states_implicit:",final_states_implicit)
    
        # Apply kernel density estimation
        kde_explicit = gaussian_kde(final_states_explicit)
        kde_implicit = gaussian_kde(final_states_implicit)
    
        # Define a range for the x-axis to generate PDF data
        x_vals = np.linspace(-4, 4, 1000)
        pdf_explicit = kde_explicit(x_vals)
        pdf_implicit = kde_implicit(x_vals)
        
    #     print("pdf_explicit:",pdf_explicit)
    #     print("pdf_implicit:",pdf_implicit)
    
        # Normalize PDF
        pdf_explicit /= trapz(pdf_explicit, x_vals)
        pdf_implicit /= trapz(pdf_implicit, x_vals)
    
        if plot_on:
            # Plot the estimated PDF
            plt.figure(figsize=(12, 8))
            plt.plot(x_vals, pdf_implicit, label=f'Implicit Euler KDE (dt={dt})', color='tab:orange', linestyle='-.')
            plt.plot(x_vals, pdf_explicit, label=f'Explicit Euler KDE (dt={dt})', color='tab:blue', linestyle=':')
            plt.xlabel('X')
            plt.ylabel('Density')
            plt.legend()
            plt.title(f'KDE Comparison of Explicit and Implicit Euler Methods at the end time T={T} (dt={dt})')
            plt.grid(True)
            figure_filename = os.path.join(add_mypathlevy.kdefigure_folder, f'PureKDE_Comparison_noise_{noise_type}_initial_type{initial_type}_atT_{T}_dt_{dt}.{figure_format.lower()}')
            plt.savefig(figure_filename, dpi=dpi)
            plt.show() 
            #plt.close()  # Close the plot to avoid pausing the script
    
        # Calculate the absolute error of the integral (IAE)
        iae = trapz(np.abs(pdf_explicit - pdf_implicit), x_vals)
        print(f' PDF Integrated Absolute Differenc (IAD) at the end time T={T} for dt={dt}:', iae)
    
        # Compare computation time
        print(f'Computation time for Explicit Euler with dt={dt}:', time_explicit, 'seconds')
        print(f'Computation time for Implicit Euler with dt={dt}:', time_implicit, 'seconds')
  
        # Save meta data
        metadata = {
            "dt": dt,
            "IAE": iae,
            "Implicit computation time": time_implicit,
            "Explicit computation time": time_explicit,
        }
        with open(os.path.join(add_mypathlevy.kdedata_folder, f"PureEulerComparison_noise_{noise_type}_initial_type{initial_type}_atT_{T}_dt_{dt}.json"), "w") as f:
            json.dump(metadata, f)
        