# Import Necessary libraries
import pandas as pd
import re
import glob
from collections import Counter
import matplotlib.pyplot as plt
import os
from srs.utils_text import clean_amharic_text



def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return pd.DataFrame({'text': lines})

