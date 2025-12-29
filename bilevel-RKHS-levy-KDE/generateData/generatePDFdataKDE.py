import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
import os
import add_mypathlevy


# KDE to estimate p(x,t)
def kde_density_estimation(paths, x_grid, bandwidth, threshold=1e-6):
    Nt = paths.shape[1]
    Nx = len(x_grid)
    densities = np.zeros((Nt, Nx))
    for i in range(Nt):
        print("Iteration steps:", i)
        samples = paths[:, i]
        std_dev = np.std(samples, ddof=1)
        if std_dev < threshold:
            return np.zeros_like(x_grid)
        kde = gaussian_kde(samples, bw_method=bandwidth / std_dev)
        densities[i, :] = kde.evaluate(x_grid)
    return densities

def genereatePDFdataKDE(paths, x_grid, bandwidth):
    """
    Generate PDF data from trajectories using KDE, 
    then apply a Savitzky-Golay filter to smooth the PDF for derivative computation.

    """
    density_datakde = kde_density_estimation(paths, x_grid, bandwidth, threshold=1e-6)
    pdf_data = savgol_filter(density_datakde, window_length=7, polyorder=3, axis=0)
    return density_datakde, pdf_data
    

# Save p(x,t) data
def save_densitydata(dt, T, x_grid, time_grid, p_val, densities, bandwidth, noise_type, initial_type, method):
    # Create a save folder
    folder = os.path.join(add_mypathlevy.kdedata_folder, f"Pure{method}_results_noise_{noise_type}_initial_type{initial_type}_dt_{dt}_T_{T}_bandwidth{bandwidth}")#_sigma80
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Save density data without filter
    np.save(os.path.join(folder, f"KDEdensity_nofilter_{method}.npy"), p_val)
    
    # Save density data with filter
    np.save(os.path.join(folder, f"KDEdensity_{method}.npy"), densities)
    
    # Save variables
    np.savez(os.path.join(folder, f"KDEparameters_{method}.npz"), dt=dt, T=T, x_grid=x_grid, time_grid=time_grid)


# Load p(x,t) data
def load_densitydata(dt, T, bandwidth, noise_type,initial_type, method):

    folder = os.path.join(add_mypathlevy.kdedata_folder, f"Pure{method}_results_noise_{noise_type}_initial_type{initial_type}_dt_{dt}_T_{T}_bandwidth{bandwidth}")
    
    # Load pdf data with filter
    Density_set = np.load(os.path.join(folder, f"KDEdensity_{method}.npy"))
    
    # Load variables
    KDEparam_set = np.load(os.path.join(folder, f"KDEparameters_{method}.npz"))
    
    return Density_set, KDEparam_set