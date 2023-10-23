import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()

plt.close('all')



def _DBH_to_height(DBH, b1, b2, const):
    H = ( DBH / (b1 + b2*DBH) )**3 + const 
    return H

def _height_to_DBH(H, b1, b2, const):
    k = np.cbrt(H - const)
    DBH = np.clip( (b1 * k) / (1 - b2*k), a_min =  0, a_max = None)       
    return DBH




# %% Spruces
## Model 1:
# https://academic.oup.com/forestry/article/95/5/634/6580516#375802885
# https://www.diva-portal.org/smash/get/diva2:1605781/FULLTEXT01.pdf
b1 = 0.9133
b2 = 0.1303
# Calculate y values 
# spruce_y1 = 0.1*_DBH_to_height(x, b1, b2, 13)

# Data file paths
data_paths = [
    "N1b_all_sep_Mosshult.csv",
    "N1b_all_sep_Lilla Norrskog.csv",
    "N1b_all_sep_Öveshult.csv",
    "N1b_all_sep_Romperöd.csv",
    "N1b_all_sep_Simontorp.csv",
]
df = pd.concat([pd.read_csv("_data/trees/data_csv/" + path, delimiter=';') for path in data_paths])
# Filter rows where X is not equal to 0
df = df[df['Th'] != 0]
# plt.scatter(df['Dbh']/10, 0.1*df['Th'], marker='o', c = 'lightgreen', s = 30, alpha=0.6) # edgecolor = 'black'



# Generate x values
x = np.linspace(0, 60, 500)  # Create an array of 100 points from 0 to

## Model 2:
# https://www.tandfonline.com/doi/full/10.1080/21580103.2014.957354
b1 = 2.2131 
b2 = 0.3046
# Calculate y values
# spruce_y2 = ((x / (b1 + b2*x))**3 + 1.3)
spruce_y2 = _DBH_to_height(x, b1, b2, 1.3)




# Create the plot
plt.figure(figsize=(8, 6))
# plt.plot(x, spruce_y1, label='Spruce - model 1 (Sweden)', color='limegreen', linewidth=2)
plt.plot(x, spruce_y2, label='Spruce - model 2 (Norway)', color='green', linewidth=2)






# %% Pines


## Model 1:
# https://www.tandfonline.com/doi/full/10.1080/21580103.2014.957354
b1 = 2.2845
b2 = 0.3318
# Calculate y values
# pine_y1 = ((x / (b1 + b2*x))**3 + 1.3)
pine_y1 = _DBH_to_height(x, b1, b2, 1.3)
plt.plot(x, pine_y1, label='Pine - model 1 (Norway)', color='blue', linewidth=2)


# %% Downy birch


## Model 1:
# https://www.tandfonline.com/doi/full/10.1080/21580103.2014.957354
b1 = 1.649
b2 = 0.373	
# Calculate y values
# birch_y1 = ((x / (b1 + b2*x))**3 + 1.3)
birch_y1 = _DBH_to_height(x, b1, b2, 1.3)
plt.plot(x, birch_y1, label='Birch - model 1 (Norway)', color='orange', linewidth=2)



# Add labels and a legend
plt.xlabel('DBH [cm]')
plt.ylabel('Height [m]')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()




""" Reverse relation """
plt.figure(figsize=(8, 6))

def plot_height_DBH(b1, b2, const, color, label):    
    asymp = (1/b2)**3 + const
    # Generate height points from 0 until the 80% of the asymptote
    height_space = np.linspace(0, 0.8 * asymp, 200)
    DBH = _height_to_DBH(height_space, b1, b2, const)
    plt.plot(height_space, DBH, linewidth=2, color = color, label = label)
    
## Spruce
b1 = 2.2131 
b2 = 0.3046
plot_height_DBH(b1, b2, 1.3, color = "green", label = "Spruce (Norway)")

## Pines
b1 = 2.2845
b2 = 0.3318
plot_height_DBH(b1, b2, 1.3, color = "blue", label = "Pine (Norway)")


## Birch
b1 = 1.649
b2 = 0.373	
plot_height_DBH(b1, b2, 1.3, color = "orange", label = "Birch (Norway)")

# Add labels and a legend
plt.xlabel('Height [m]')
plt.ylabel('DBH [cm]')
plt.legend()



























