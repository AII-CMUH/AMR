import numpy as np
import pandas as pd
from pathlib import Path

rootpath = Path('path/to/file')
df = pd.read_csv(rootpath/'MRSA2019.csv')
d = pd.DataFrame(np.concatenate([np.array([df['Accession #'].unique()]), np.array([np.ones(df['Accession #'].nunique())])], 0).T, columns=['accnum', 'MRSA'])
