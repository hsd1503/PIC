import pandas as pd
import numpy as np
from tqdm import tqdm

def prism_iii(df):
    out = []
    n_row = df.shape[0]
    # for i in tqdm(range(n_row), desc='prism_iii'):
    for i in range(n_row):
        out.append(prism_iii_row(df.iloc[i]))
    out = np.array(out)
    return out

def prism_iii_row(row):
    """    
    age_month
    sbp, chart_1016_min
    hr, chart_1003_max
    temperature, chart_1001
    [not exist] pupillary reflexes, GCS

    tCO2, lab_5256_max
    pH, lab_5237_max, lab_5238_max
    PaO2, lab_5239_min, lab_5244_min
    PCO2, lab_5235_max, lab_5236_max

    glucose, lab_5223_max, lab_5047_max
    potassium, lab_5226_max, lab_5056_max
    creatinine, lab_5041_max, lab_5032_max
    bun, lab_5033_max

    wbc, lab_5141_min
    pt, lab_5186_max
    pc, lab_5129_min

    """
    MAX_SCORE = 74

    ['PID', 'age', 'SBPmin', 'SBPmax', 'Tempmin', 'Tempmax', 'Respmin', 'Respmax', 'ABEmin', 'ABEmax', 'Lacmin',
     'Lacmax', 'SBEmin', 'SBEmax', 'pCO2', 'pO2', 'K', 'HCO3', 'sO2', 'PC', 'PCT', 'Glu', 'SBC', 'M_label']
    age = row['age']
    sbp = row['SBPmin']
    #hr = row['chart_1003_max']
    temperature_max = row['Tempmax']
    temperature_min = row['Tempmin']
    #tCO2 = row['lab_5256_max']
    #pH = row['lab_5237_max'] if np.isnan(row['lab_5237_max']) else row['lab_5238_max']
    PaO2 = row['pO2']
    PCO2 = row['pCO2']
    glucose = row['Glu']
    potassium = row['K']
    #creatinine = row['lab_5041_max'] if np.isnan(row['lab_5041_max']) else row['lab_5032_max']
    #bun = row['lab_5033_max']
    wbc = row['SBC']
    pt = row['PC']
    pc = row['PC']

    if age < 1:
        patient_type = 'infant'
    elif 1 <= age < 12:
        patient_type = 'child'
    elif age >= 12:
        patient_type = 'adolescent'

    score = 0

    # vital signs
    if not np.isnan(sbp):
        if patient_type == 'infant':
            if sbp < 45:
                score += 7
            elif 45 <= sbp <= 65:
                score += 3
        elif patient_type == 'child':
            if sbp < 55:
                score += 7
            elif 55 <= sbp <= 75:
                score += 3
        elif patient_type == 'adolescent':
            if sbp < 65:
                score += 7
            elif 65 <= sbp <= 85:
                score += 3

    # if not np.isnan(hr):
    #     if patient_type == 'neonate' or patient_type == 'infant':
    #         if hr >= 225:
    #             score += 4
    #         elif 215 <= hr <= 225:
    #             score += 3
    #     elif patient_type == 'child':
    #         if hr >= 205:
    #             score += 4
    #         elif 185 <= hr <= 205:
    #             score += 3
    #     elif patient_type == 'adolescent':
    #         if hr >= 155:
    #             score += 4
    #         elif 145 <= hr <= 155:
    #             score += 3
    score += 3.5

    if not np.isnan(temperature_max) or not np.isnan(temperature_min):
        if temperature_min < 33 or temperature_max > 40:
            score += 3

    # acid-base/blood gases
    # if not np.isnan(tCO2):
    #     if tCO2 > 34:
    #         score += 4
    score += 2

    # if not np.isnan(pH):
    #     if pH > 7.55:
    #         score += 3
    #     elif 7.48 <= pH <= 7.55:
    #         score += 2
    score += 2.5

    if not np.isnan(PaO2):
        if PaO2 < 42:
            score += 6
        elif 42 <= PaO2 <= 49.9:
            score += 3

    if not np.isnan(PCO2):
        if PCO2 > 75:
            score += 3
        elif 50 <= PCO2 <= 75:
            score += 1

    # chemistry tests
    if not np.isnan(glucose):
        if glucose > 11:
            score += 2

    if not np.isnan(potassium):
        if potassium > 6.9:
            score += 3

    # if not np.isnan(creatinine):
    #     if patient_type == 'neonate':
    #         if creatinine > 75:
    #             score += 2
    #     elif patient_type == 'infant' or patient_type == 'child':
    #         if creatinine > 80:
    #             score += 2
    #     elif patient_type == 'adolescent':
    #         if creatinine > 115:
    #             score += 2
    score += 1

    # if not np.isnan(bun):
    #     if patient_type == 'neonate':
    #         if bun > 4.3:
    #             score += 3
    #     else:
    #         if bun > 5.4:
    #             score += 3

    score += 1.5

    # hematology tests
    if not np.isnan(wbc):
        if wbc < 3000:
            score += 4

    if not np.isnan(pt):
        if pt > 22:
            score += 3

    if not np.isnan(pc):
        pc = pc*1000
        if pc < 50000:
            score += 5
        elif 50000 <= pc < 100000:
            score += 4
        elif 100000 <= pc < 200000:
            score += 2
        
    return score/MAX_SCORE
        
        
        
        