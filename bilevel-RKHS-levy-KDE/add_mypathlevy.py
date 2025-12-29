import os

# Save data to a local directory
# Select the save path according to the operating system
if os.name == 'nt': # Windows
    save_dir = os.path.join(os.getenv('USERPROFILE'), 'FDMDARKHS-data', 'output')
else:  # Unix/Linux/MacOS
    save_dir = os.path.join(os.getenv('HOME'), 'FDMDARKHS-data', 'output')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
 
# Ensure that the directory where the data is stored exists
data_folder = os.path.join(save_dir, 'Dataset')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    
# Ensure that the subdirectory where the image is saved exists
figure_folder = os.path.join(save_dir, 'Figures')
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

##########============================== KDE =======================##################
 
# Ensure that the directory where the data is stored exists
kdedata_folder = os.path.join(save_dir, 'PurejumpKDEDataset')
if not os.path.exists(kdedata_folder):
    os.makedirs(kdedata_folder)
        
# Ensure that the subdirectory where the image is saved exists
kdefigure_folder = os.path.join(save_dir, 'KDEFigures')
if not os.path.exists(kdefigure_folder):
    os.makedirs(kdefigure_folder)