import os
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import requests
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np
import pprint
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.stats.outliers_influence import variance_inflation_factor as get_vif
import ml.shared as shared


shared.use_cpu_and_make_results_reproducible()