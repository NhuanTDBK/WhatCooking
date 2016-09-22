from os import listdir
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
folder_name = 'results/'
files = listdir('results')
submissions = [pd.read_csv(folder_name+item) for item in files]
encoder = LabelEncoder()
last_col = [submissions[0].columns[-1]]
encoder.fit(submissions[0][last_col])
a = zip(*[encoder.transform(sub[last_col]) for sub in submissions])
b = [Counter(item).most_common(1)[0][0] for item in np.array(a).reshape(-1,len(files))]
y_sub = encoder.inverse_transform(b)
from utils import *
save_submission("Vote2",0.77,y_sub)
