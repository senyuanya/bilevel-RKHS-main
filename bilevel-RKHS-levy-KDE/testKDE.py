import numpy as np
import random
import pandas as pd
import sys
import matplotlib.pyplot as plt
from generateData.generateDataKDE import generateDataKDE, compute_nonlocal_term_symmetric_convolution
from stabilize_matricesKDE import stabilize_matricesKDE

from regression.getData4regression_auto import getData4regression_auto

from iDARR.auto_kernel_mat import auto_kernel_mat  
from iDARR.Lcurve import Lcurve  
from iDARR.idarr import idarr   
from regularization.Tikh_Lcurve import Tikh_Lcurve   
from regularization.Tikh_auto_basis_Lcurve import Tikh_auto_basis_Lcurve
from regularization.Tikh_biSGDGbarEarlyStop import Tikh_auto_basis_SGD_nesterov_normalized
from regularization.Tikh_GCV import Tikh_GCV  
from regularization.Tikh_auto_basis_GCV  import Tikh_auto_basis_GCV


# Set the random seed for reproducibility
random.seed(1)
np.random.seed(1)

######==================== Generate the PDF Data with Finite Difference Method ======================== #######
## 0. setttings-----------------------------------------------------------------
# Parameters
#Set parameters
X0 = 0 # initial value
N = 30#14 # number of data pairs (u_i, f_i);
u_Type = 'Compoundpdf'  # types: 'Compoundpdf';'Laplacepdf'
x_range = [-5, 5] # space range
dt = 0.05#5 # time mesh size
t_range = [0, 5] # time range
example_type = 'nonlocal'  # {'LinearIntOpt', 'nonlocal', 'mfOpt'}
kernel_type = 'Compoundlevy'  # Gaussian, sinkx, FracLap,Compoundlevy,Laplacejump
r0 = 0.05 #cutoff distance of singular levy kernel
bandwidth = 0.105 #0.105#0.21#0.021
initial_type = "zero" #'zero'；piecewise
######==================== Generate the PDF Data with Finite Difference Method ======================== #######
obsInfo, kernelInfo, ux_valtr, fx_valtr, ux_valts, fx_valts = generateDataKDE(t_range, dt, N, r0, bandwidth, kernel_type, initial_type, example_type)

# get boundary width and x-mesh in use. --------  
dx = obsInfo['x_mesh_dx'] 
x_mesh = obsInfo['x_mesh_data']
bdry_width = int(np.floor((obsInfo['delta']  + 1e-10) / dx))
r_seq = dx * np.arange(1, bdry_width + 1)
Index_xi_inUse = np.arange(bdry_width, len(obsInfo['x_mesh_data']) - bdry_width)

## 2. pre-process data, get all data elements for regression:   ****** a key step significantly reducing computational cost-----------------------------------------------------------------
normalizeOn = True #obsInfo['example_type'] != 'classicalReg'
fun_g_vec = obsInfo['fun_g_vec']
data_str = f"{obsInfo['example_type']}{kernelInfo['kernel_type']}{u_Type}{obsInfo['x_mesh_str']}_dx_coarse{dx:.4f}_"
regressionData = getData4regression_auto(ux_valtr, fx_valtr, ux_valts, fx_valts, dx, obsInfo, bdry_width, Index_xi_inUse, r_seq, data_str, normalizeOn)

rho_val = regressionData['rho_val']
ind_rho = np.where(rho_val > 0)[0]
rho_val = rho_val[ind_rho]
r_seq = regressionData['r_seq'][ind_rho] # when r_seq is non-uniform, use dr = r_seq(2:end) - r_seq(1:end-1). 

dr = r_seq[1] - r_seq[0]

# Get true kernel values
K_true = kernelInfo['K_true']
K_true_val = K_true(r_seq)

## 3. Iterative regularizations -----------------------------------------------------------------
# compute auto-regularized solution
K = 50
G_D1, basis_D1, Sigma_D1 = auto_kernel_mat(regressionData, 'auto')
X1, res1, eta1, iter_stop1 = idarr(regressionData, 'auto', K, 'LC', 0)
k1, _ = Lcurve(res1, eta1, 111, 'iDarr, auto-RKHS')
x_reg1 = X1[:, iter_stop1]

# compute Gbar-RKHS solution
g = regressionData['g_ukxj']
ns, J, n0 = g.shape
k = n0 * J
g1 = np.zeros((ns, k))
for i in range(n0):
    g1[:, i*J:(i+1)*J] = g[:, :, i]

fx_vec = regressionData['fx_vec'].T  # Transpose, J x n0
f = regressionData['fx_vec'].flatten()  # Flatten to vector
A = g1.T @ G_D1 * dr

# Bilevel-SGD-RKHS
##Set the validation data
# compute Gbar-RKHS solution
g_ts = regressionData['g_ukxj_ts']
ns_ts, J_ts, n0_ts = g_ts.shape
k_ts = n0_ts * J_ts
g1_ts = np.zeros((ns_ts, k_ts))
for i in range(n0_ts):
    g1_ts[:, i*J_ts:(i+1)*J_ts] = g_ts[:, :, i]
    
fx_vec_ts = regressionData['fx_vec_ts'].T  # Transpose, J x n0
f_ts = regressionData['fx_vec_ts'].flatten()  # Flatten to vector
A_ts = g1_ts.T @ G_D1 * dr    

######==================== Stabilize input matrices A and G_D1 for CGSVD ======================== #######
A_stable, A_stable_ts, G_D1_stable, info = stabilize_matricesKDE(A, A_ts, G_D1)

#==============================
method = 'Tikh'  # method can be 'tsvd', 'Tikh', 'dsvd'
c_reg2, res2, eta2, reg_corner = Tikh_Lcurve(A_stable, f, G_D1_stable, method)
x_reg2 = G_D1_stable @ c_reg2

# compute Gaussian-RKHS solution
X3, res3, eta3, iter_stop3 = idarr(regressionData, 'gauss', K, 'LC', 0, 0.01)
k3, _ = Lcurve(res3, eta3, 3, 'iDarr, Gaussian-RKHS')
x_reg3 = X3[:, iter_stop3]

## 4. Tikhonov regularization with L-curve -----------------------------------------------------------------
# auto-basis funcitons
x_reg4, res4, eta4, reg4_corner= Tikh_auto_basis_Lcurve(regressionData, 'auto-RKHS')
x_reg5, res5, eta5, reg5_corner = Tikh_auto_basis_Lcurve(regressionData, 'Gaussian-RKHS', 0.01)  # Gaussian-RKHS

# Gbar-RKHS
method = 'Tikh'  # method can be 'tsvd', 'Tikh', 'dsvd'
c_reg6, res6, eta6, reg6_corner = Tikh_Lcurve(A_stable, f, G_D1_stable, method)
x_reg6 = G_D1_stable @ c_reg6

c_reg8, G8, reg8_corner = Tikh_GCV(A_stable, f, G_D1_stable, method)
x_reg8 = G_D1_stable @ c_reg8

x_reg9, res9, eta9, reg9_corner= Tikh_auto_basis_GCV(regressionData, 'auto-RKHS')

# Bilevel-SGD-RKHS    
method_our = 'BiSNGD' 
scale_value = 1 # scale_value = 500 if Compoundpdf. scale_value = 0.5 if pureStablepdf
# rho_val = regressionData['rho_val']
c_reg7, res7, eta7, reg_corner7, loss_history, gamma_history, iteration_history, xreg7_history, gradient_norm_history, gamma_update_direction,D_matrix = Tikh_auto_basis_SGD_nesterov_normalized(A_stable, f, A_stable_ts, f_ts, G_D1_stable, scale_value, rho_val, method_our)
x_reg7 = G_D1_stable @ c_reg7
##compute the error
#erro results
error_history = [ np.sqrt(dr * rho_val.T @ (((G_D1_stable @ x7).ravel() - K_true_val.T) ** 2)) for x7 in xreg7_history]
log10_loss = [np.log10(loss) for loss in loss_history]
log10_error = [np.log10(error) for error in error_history]
## SGD loss -------------------------------------
# Plot loss over iterations
plt.figure()
plt.semilogy(iteration_history, loss_history, label='loss')
plt.semilogy(iteration_history, error_history, label='error')
plt.xlabel('Iteration')
plt.ylabel('Loss and error')
plt.title('Loss over SGD Iterations')
plt.legend()
plt.show()

plt.figure()
plt.plot(iteration_history[0:1000], loss_history[0:1000], label='loss')
plt.plot(iteration_history[0:1000], error_history[0:1000], label='error')
plt.xlabel('Iteration')
plt.ylabel('Loss and error')
plt.title('Loss over SGD Iterations')
plt.legend()
plt.show()

plt.figure()
plt.plot(iteration_history, loss_history, label='loss')
plt.xlabel('Iteration')
plt.ylabel('Loss and error')
plt.title('Loss over SGD Iterations')
plt.legend()
plt.show()
   
   
# Plot gamma over iterations
plt.figure()
plt.plot(iteration_history, gamma_history)
plt.xlabel('Iteration')
plt.ylabel('Gamma')
plt.title('Gamma over SGD Iterations')
plt.show()

# Plot gamma over iterations
plt.figure()
plt.plot(iteration_history, gamma_history)
plt.plot(iteration_history, error_history, label='error')
plt.xlabel('Iteration')
plt.ylabel('Gamma and Error')
plt.title('Gamma and error over SGD Iterations')
plt.show()

# relative error
er1 = np.zeros(len(eta1))
er3 = np.zeros(len(eta3))
xx = K_true_val.T
nx = np.sqrt(dr * rho_val.T @ (xx ** 2))  # norm(xx)

for i in range(len(eta1)):
    er1[i] = np.sqrt(dr * rho_val.T @ ((X1[:, i] - xx) ** 2))  # norm(X1[:,i]-xx) / nx

for i in range(len(eta3)):
    er3[i] = np.sqrt(dr * rho_val.T @ ((X3[:, i] - xx) ** 2))  # norm(X3[:,i]-xx) / nx

er2 = np.sqrt(dr * rho_val.T @ ((x_reg2.ravel() - xx) ** 2))  # norm(x_reg2-xx) / nx
er6 = np.sqrt(dr * rho_val.T @ ((x_reg6.ravel() - xx) ** 2))  # norm(x_reg6-xx) / nx
er7 = np.sqrt(dr * rho_val.T @ ((x_reg7.ravel() - xx) ** 2))  # norm(x_reg5-xx) / nx
er4 = np.sqrt(dr * rho_val.T @ ((x_reg4.ravel() - xx) ** 2))  # norm(x_reg4-xx) / nx
er5 = np.sqrt(dr * rho_val.T @ ((x_reg5.ravel() - xx) ** 2))  # norm(x_reg5-xx) / nx
er8 = np.sqrt(dr * rho_val.T @ ((x_reg8.ravel() - xx) ** 2))  # norm(x_reg5-xx) / nx
er9 = np.sqrt(dr * rho_val.T @ ((x_reg9.ravel() - xx) ** 2))  # norm(x_reg4-xx) / nx
# select optimal k for iterative methods
k1_opt = np.argmin(er1)
k3_opt = np.argmin(er3)

## Compare estimators  ------------------------------------------------------------
print('Relative L2rho Errors: \n')
methods_all = ["auto-RKHS-iDarr", "Gbar-Tikh", "Gaussian-iDarr", "auto-RKHS-LC", "Gaussian-RKHS-LC", "Gbar-LC", "Gbar-GCV", "auto-RKHS-GCV", "Gbar-Bilevel"]
rel_err = np.array([er1[k1_opt], er2, er3[k3_opt], er4, er5, er6,er8, er9, er7])
relative_err = pd.DataFrame({'Methods': methods_all, 'Relative Error': rel_err})
print(relative_err)

# Plot estimators
plt.figure(11, dpi=500)
plt.plot(r_seq, K_true_val, 'k:', linewidth=3, label='True')
plt.plot(r_seq, x_reg1, '-.', linewidth=2, label='auto-RKHS')
plt.plot(r_seq, x_reg2, '-.', linewidth=2, label='Tikh Gbar-RKHS')
plt.plot(r_seq, x_reg3, '-.', linewidth=2, label='Gaussian-RKHS')
plt.legend()
plt.title('Iterative Estimators, L-curve')

plt.figure(12, dpi=500)
plt.plot(r_seq, K_true_val, 'k:', linewidth=3, label='True')
plt.plot(r_seq, X1[:, k1_opt], '-.', linewidth=2, label='auto-RKHS')
plt.plot(r_seq, x_reg2, '-.', linewidth=2, label='Tikh Gbar-RKHS')
plt.plot(r_seq, X3[:, k3_opt], '-.', linewidth=2, label='Gaussian-RKHS')
plt.legend()
plt.title('Iterative Estimators, opt')

plt.figure(13, dpi=500)
plt.semilogy(range(1, K+1), er2*np.ones(K), 'k-', linewidth=3.0, label='Tikh-LC')
plt.semilogy(range(1, len(eta1)+1), er1, 'bo-', linewidth=2.0, label='auto-RKHS')
plt.semilogy(range(1, len(eta3)+1), er3, 'm>-', linewidth=2.0, label='Gaussian-RKHS')
plt.xlabel('Iteration', fontsize=16)
plt.legend(fontsize=12, loc='lower right')
plt.ylabel('Relative error', fontsize=16)
plt.grid(True, which='both', alpha=0.3)
plt.minorticks_on()
plt.grid(which='minor', alpha=0.01)

plt.figure(14, dpi=500)
plt.plot(r_seq, K_true_val, 'k:', linewidth=3, label='True')
plt.plot(r_seq, x_reg6, 'b-', linewidth=2, label='Gbar-RKHS')
plt.plot(r_seq, x_reg4, '-.', linewidth=2, label='auto-RKHS')
plt.plot(r_seq, x_reg5, '-.', linewidth=2, label='Gaussian-RKHS')
plt.legend()
plt.title('Tikhonov Estimators, L-curve')

plt.figure(15, dpi=500); plt.clf()
plt.plot(r_seq, x_reg2, 'c-x', linewidth=1, label='Gbar-RKHS-Tikh')
plt.plot(r_seq, x_reg1, 'c:o', linewidth=1, label='auto-RKHS-iDarr')
plt.plot(r_seq, x_reg6, 'b-', linewidth=2, label='Gbar-RKHS')
plt.plot(r_seq, x_reg4, '-.', linewidth=2, label='auto-RKHS')
plt.plot(r_seq, x_reg5, '-.', linewidth=2, label='Gaussian-RKHS')
plt.plot(r_seq, K_true_val, 'k:', linewidth=3, label='True')
plt.legend()
plt.title('Estimators: TSVD, iDarr and L-curves')

plt.figure(16, dpi=500); plt.clf()
plt.plot(r_seq, x_reg6, 'b-', linewidth=2, label='Gbar-RKHS')
plt.plot(r_seq, x_reg4, '-.', linewidth=2, label='auto-RKHS')
plt.plot(r_seq, K_true_val, 'k:', linewidth=3, label='True')
plt.legend()
plt.title('Estimators: Gbar-RKHS and auto-RKHS')

plt.figure(17, dpi=500); plt.clf()
plt.plot(r_seq, x_reg6, 'b-', linewidth=2, label='Gbar-RKHS')
plt.plot(r_seq, K_true_val, 'k:', linewidth=3, label='True')
plt.legend()
plt.title('Estimators: Gbar-RKHS')

plt.figure(18, dpi=500); plt.clf()
plt.plot(r_seq, x_reg4, '-.', linewidth=2, label='auto-RKHS')
# plt.plot(r_seq, rho_val*200, 'r-', linewidth=2, label='Gbar-RKHS')
plt.plot(r_seq, K_true_val, 'k:', linewidth=3, label='True')
plt.legend()
plt.title('Estimators: auto-RKHS')

plt.figure(19, dpi=500); plt.clf()
plt.plot(r_seq, x_reg7, '-.', linewidth=2, label='SGD-RKHS')
plt.plot(r_seq, K_true_val, 'k:', linewidth=3, label='True')
plt.legend()
plt.title('Estimators: BilevelSGD-RKHS')

plt.figure(110, dpi=500); plt.clf()
plt.plot(r_seq, x_reg6, 'b-', linewidth=2, label='Gbar-RKHS')
#plt.plot(r_seq, x_reg8, '-.', linewidth=2, label='Gbar-GCV-RKHS')
plt.plot(r_seq, x_reg9, '-.', linewidth=2, label='auto-RKHS-GCV')
plt.plot(r_seq, x_reg4, '-.', linewidth=2, label='auto-RKHS')
plt.plot(r_seq, x_reg7, '-.', linewidth=2, label='SGD-RKHS')
plt.plot(r_seq, K_true_val, 'k:', linewidth=3, label='True')
plt.legend()
plt.title('Estimators: Gbar-RKHS, auto-RKHS and BilevelSGD')

plt.show()

# ## ============================= L2 norm and l2 norm =========================
# from regularization.Tikh_auto_basis_SGD_nesterov_norm import Tikh_auto_basis_SGD_nesterov_norm
# # --------------------L_2 norm and l_2 norm---------------------------------------------------------
# # ||\phi||_{L_2}^2 = c^T K_rho c where K_rho[i,j] = ∫ \bar{G}(r_i, r) \bar{G}(r_j, r) ρ(r) dr
# regressionData['L2'] = G_D1_stable @ np.diag(rho_val) @ G_D1_stable.T * dx
# # ||phi||_{l^2}^2 = c^T * Gbar * Gbar^T * c,
# regressionData['l2'] =  np.eye(*G_D1_stable.shape) 


# L2 = regressionData['L2']
# l2 = regressionData['l2']
# # Gbar-RKHS
# method = 'Tikh'  # method can be 'tsvd', 'Tikh', 'dsvd'
# c_reg6L, res6L, eta6L, reg6_cornerL = Tikh_Lcurve(A_stable, f, L2, method)
# x_reg6L = G_D1_stable @ c_reg6L

# c_reg6l, res6l, eta6l, reg6_cornerl = Tikh_Lcurve(A_stable, f, l2, method)
# x_reg6l = G_D1_stable @ c_reg6l

# # GCV
# c_reg8L, G8L, reg8_cornerL = Tikh_GCV(A_stable, f, L2, method)
# x_reg8L = G_D1_stable @ c_reg8L

# c_reg8l, G8l, reg8_cornerl = Tikh_GCV(A_stable, f, l2, method)
# x_reg8l = G_D1_stable @ c_reg8l

# # Bi-level
# scale_valueL = 1
# method_our = 'BiSNGD' 
# c_reg7L, res7L, eta7L, reg_corner7L, loss_historyL, gamma_historyL, iteration_historyL, xreg7_historyL = Tikh_auto_basis_SGD_nesterov_norm(A_stable, f, A_stable_ts, f_ts, L2, scale_valueL, rho_val, method_our)
# x_reg7L = G_D1_stable @ c_reg7L

# error_historyL = [ np.sqrt(dr * rho_val.T @ (((G_D1_stable @ x7_L).ravel() - K_true_val.T) ** 2)) for x7_L in xreg7_historyL]
# # Plot loss over iterations
# plt.figure()
# plt.semilogy(iteration_historyL , loss_historyL , label='loss')
# plt.semilogy(iteration_historyL , error_historyL , label='Error')
# plt.xlabel('Iteration')
# plt.ylabel('Loss and error')
# plt.title('Loss over SGD Iterations')
# plt.legend()
# plt.show()

# # Plot gamma over iterations
# plt.figure()
# plt.plot(iteration_historyL, gamma_historyL)
# plt.xlabel('Iteration')
# plt.ylabel('Gamma')
# plt.title('Gamma over SGD Iterations')
# plt.show()

# scale_valuel = 1
# c_reg7l, res7l, eta7l, reg_corner7l, loss_historyl, gamma_historyl, iteration_historyl, xreg7_historyl = Tikh_auto_basis_SGD_nesterov_norm(A_stable, f, A_stable_ts, f_ts, l2, scale_valuel, rho_val, method_our)
# x_reg7l = G_D1_stable @ c_reg7l

# error_historyl = [ np.sqrt(dr * rho_val.T @ (((G_D1_stable @ x7_l).ravel() - K_true_val.T) ** 2)) for x7_l in xreg7_historyl]

# plt.figure()
# plt.semilogy(iteration_historyl, loss_historyl, label='loss')
# plt.semilogy(iteration_historyl, error_historyl, label='Error')
# plt.xlabel('Iteration')
# plt.ylabel('Loss and error')
# plt.title('Loss over SGD Iterations')
# plt.legend()
# plt.show()

# # Plot gamma over iterations
# plt.figure()
# plt.plot(iteration_historyl, gamma_historyl)
# plt.xlabel('Iteration')
# plt.ylabel('Gamma')
# plt.title('Gamma over SGD Iterations')
# plt.show()

# er6L = np.sqrt(dr * rho_val.T @ ((x_reg6L.ravel() - xx) ** 2)) 
# er6l = np.sqrt(dr * rho_val.T @ ((x_reg6l.ravel() - xx) ** 2)) 
# er8L = np.sqrt(dr * rho_val.T @ ((x_reg8L.ravel() - xx) ** 2)) 
# er8l = np.sqrt(dr * rho_val.T @ ((x_reg8l.ravel() - xx) ** 2)) 
# er7L = np.sqrt(dr * rho_val.T @ ((x_reg7L.ravel() - xx) ** 2)) 
# er7l = np.sqrt(dr * rho_val.T @ ((x_reg7l.ravel() - xx) ** 2)) 

# ## Compare Norm  ------------------------------------------------------------
# print('Relative L2rho Errors with Different Norm: \n')
# methods_allL = [ "Gbar-LC-RKHS", "Gbar-GCV-RKHS", "Gbar-Bilevel-RKHS", "Gbar-LC-L2", "Gbar-LC-l2", "Gbar-GCV-L2", "Gbar-GCV-l2", "Gbar-Bilevel-L2", "Gbar-Bilevel-l2"]
# rel_errL = np.array([er6,er8, er7, er6L, er6l, er8L, er8l, er7L, er7l])
# relative_errL = pd.DataFrame({'Methods': methods_allL, 'Relative Error': rel_errL})
# print(relative_errL)

# plt.figure(111, dpi=500, figsize=(24, 18)) 
# plt.clf()

# # RKHS-based methods
# plt.plot(r_seq, x_reg6, color='blue',  linestyle='-',  linewidth=8, label='Gbar-RKHS')
# plt.plot(r_seq, x_reg8, color='red',   linestyle='--', linewidth=8, label='Gbar-GCV-RKHS')
# plt.plot(r_seq, x_reg4, color='green', linestyle='-.', linewidth=8, label='auto-RKHS')
# plt.plot(r_seq, x_reg7, color='purple',linestyle=':',  linewidth=8, label='SGD-RKHS')

# # L2-based methods
# plt.plot(r_seq, x_reg6L, color='cyan',     linestyle='-',  linewidth=8, label='Gbar-L2')
# plt.plot(r_seq, x_reg8L, color='magenta',  linestyle='--', linewidth=8, label='Gbar-GCV-L2')
# plt.plot(r_seq, x_reg7L, color='orange',   linestyle='-.', linewidth=8, label='SGD-L2')

# # l2-based methods
# plt.plot(r_seq, x_reg6l, color='brown',  linestyle='-',  linewidth=8, label='Gbar-l2')
# plt.plot(r_seq, x_reg8l, color='black',  linestyle='--', linewidth=8, label='Gbar-GCV-l2')
# plt.plot(r_seq, x_reg7l, color='teal',   linestyle=':',  linewidth=8, label='SGD-l2')

# # True solution
# plt.plot(r_seq, K_true_val, color='k', linestyle=':', linewidth=10, label='True')

# # Set axis labels with increased font size
# plt.xlabel("r", fontsize=24)
# plt.ylabel("Value", fontsize=24)

# # Optionally, adjust tick label sizes explicitly
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)

# plt.legend(fontsize=20)
# plt.show()


# plt.figure(112, dpi=500, figsize=(24, 18)) 
# plt.clf()

# # RKHS-based methods
# plt.plot(r_seq, x_reg7, color='purple',linestyle=':',  linewidth=8, label='SGD-RKHS')

# # L2-based methods
# plt.plot(r_seq, x_reg7L, color='orange',   linestyle='-.', linewidth=8, label='SGD-L2')

# # l2-based methods
# plt.plot(r_seq, x_reg7l, color='teal',   linestyle=':',  linewidth=8, label='SGD-l2')

# # True solution
# plt.plot(r_seq, K_true_val, color='k', linestyle=':', linewidth=10, label='True')

# # Set axis labels with increased font size
# plt.xlabel("r", fontsize=24)
# plt.ylabel("Value", fontsize=24)

# # Optionally, adjust tick label sizes explicitly
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)

# plt.legend(fontsize=20)
# plt.show()

# # # # # # # # # # # # # ### Save data for picture------------------------
# import add_mypathlevy
# import os
# datafolder = os.path.join(add_mypathlevy.kdedata_folder, f"KDEpure{method_our}_dx{np.around(dx, decimals=2)}_dt{dt}_N{N}_supp_u{obsInfo['supp_H'][-1]}_t_range{t_range[-1]}_kernel_type{kernel_type}_initial_type{initial_type}_bandwidth{bandwidth}")
# if not os.path.exists(datafolder):
#     os.makedirs(datafolder)
# np.savez(os.path.join(datafolder, f"bikernel_data_scale{scale_value}.npz"), r_seq=r_seq, x_reg1=x_reg1, x_reg1_opt=X1[:, k1_opt], x_reg2=x_reg2, x_reg3=x_reg3, x_reg3_opt=X3[:, k1_opt], x_reg4 = x_reg4, x_reg5 = x_reg5,  x_reg6=x_reg6, x_reg7=x_reg7, x_reg8=x_reg8, x_reg9=x_reg9, K_true_val = K_true_val, reg_corner =reg_corner, reg4_corner= reg4_corner, reg6_corner=reg6_corner, reg7_corner=reg_corner7, reg8_corner=reg8_corner,reg9_corner=reg9_corner)
# np.savez(os.path.join(datafolder, f"bikernel_loss_scale{scale_value}.npz"), iteration_history=iteration_history, loss_history=loss_history, error_history=error_history, gamma_history = gamma_history, relative_err = relative_err,  relative_errL =relative_errL)
# np.savez(os.path.join(datafolder, f"bikernel_Gbar_scale{scale_value}.npz"), G_D1=G_D1, rho_val = rho_val, xreg7_history = xreg7_history, A=A, x_mesh = obsInfo['x_mesh_data'])
# np.savez(os.path.join(datafolder, f"bikernel_L2l2{scale_value}.npz"), x_reg6L=x_reg6L, x_reg7L=x_reg7L, x_reg8L=x_reg8L, x_reg6l=x_reg6l, x_reg7l=x_reg7l, x_reg8l=x_reg8l, reg6_cornerL= reg6_cornerL, reg6_cornerl = reg6_cornerl, reg7_cornerL=reg_corner7L, reg7_cornerl=reg_corner7l, reg8_cornerL = reg8_cornerL, reg8_cornerl = reg8_cornerl)
