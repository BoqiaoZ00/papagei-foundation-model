import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent  # Adjust based on your file structure
csv_path = project_root / "data" / "kaggle" / "MI_PPG.csv"
data_csv = pd.read_csv(csv_path)
data = data_csv.values
print(data)
plt.plot(data[0, 0:-1])
plt.show()