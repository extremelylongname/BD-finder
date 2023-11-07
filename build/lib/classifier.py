import os
import sys
import pandas as pd
import numpy as np
import torch
import sklearn as sk
import astropy
from joblib import load

def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    data_file = os.path.join(data_dir, 'allmodels.csv')
    data = pd.read_csv(data_file)
    return data
    
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'ml-models')
    model_paths = {
        'ann': os.path.join(models_dir, 'bd_ann.pt'),
        'nay': os.path.join(models_dir, 'nay.joblib'),
        'naytree': os.path.join(models_dir, 'naytree.joblib'),
        'otypeclassif': os.path.join(models_dir, 'otypeclassif.joblib'),
    }

    loaded_models = {}
    for model_name, model_path in model_paths.items()[1:4]:
        loaded_model = load(model_path)
        loaded_models[model_name] = loaded_model
    
    loaded_models.append(os.path.join(models_dir, 'bd205spectraltype.pth'))
    return loaded_models

def EnsembleClassifier(df):
    """
    Arguments:
    
    Pandas dataframe with the following columns (in AB magnitude):
    Ymag
    Jmag
    Hmag
    Kmag
    W1mag
    W2mag
    W3mag
    
    Returns:
    
    ML-based classification on if an object is a T or Y dwarf
    Physical properties of detected dwarf
    """
    dfcombined = load_data()
    models = load_models()
    nay, naytree, otypeclassif = models[0], models[1], models[2]
    
    #setting up dataframe
    print("Setting up dataframe.")
    df['otype_classified'] = None
    df['sptype_classified'] = None
    df['Mass'] = None
    df['Age'] = None
    df['Temp'] = None
    df['logg'] = None
    df['Metallicity'] = None
    
    if 'Y-J' not in df.columns:
        print("Warning: Y-J color columns are not specified. Colors will be calculated by subtracting given magnitudes.")
        df['Y-J']=df['Ymag']-df['Jmag']
    if 'J-H' not in df.columns:
        print("Warning: J-H color columns are not specified. Colors will be calculated by subtracting given magnitudes.")
        df['J-H']=df['Jmag']-df['Hmag']
    if 'H-K' not in df.columns:
        print("Warning: H-K color columns are not specified. Colors will be calculated by subtracting given magnitudes.")
        df['H-K']=df['Hmag']-df['Kmag']
    if 'J-K' not in df.columns:
        print("Warning: J-K color columns are not specified. Colors will be calculated by subtracting given magnitudes.")
        df['J-K']=df['Jmag']-df['Kmag']
    if 'W1-W2' not in df.columns:
        print("Warning: W1-W2 color columns are not specified. Colors will be calculated by subtracting given magnitudes.")
        df['W1-W2']=df['W1mag']-df['W2mag']
    if 'W2-W3' not in df.columns:
        print("Warning: W2-W3 color columns are not specified. Colors will be calculated by subtracting given magnitudes.")
        df['W2-W3']=df['W2mag']-df['W3mag']
    
    print("Starting classification. May take a long time depending on dataset length.")
    for i in range(0, len(df)):
        otype_pred1 = otypeclassif.predict(df.iloc[i][['Y-J', 'J-H', 'H-K', 'J-K']].to_numpy().reshape(1, -1))
        #object-type classification
        if otype_pred1:
            if 'W1mag' in df.columns.values:
                df.iloc[[i], df.columns.get_loc('otype_classified')] = 'substellar'
                
                spt_pred_knn = round(float(nay.predict(df.iloc[i][['Ymag', 'Jmag', 'Hmag', 'Kmag', 'W1mag', 'W2mag', 'W3mag', 'Y-J', 'J-H', 'H-K', 'J-K', 'W1-W2', 'W2-W3']].to_numpy().reshape(1, -1))[0]), 1)
                spt_pred_tree = round(float(naytree.predict(df.iloc[i][['Ymag', 'Jmag', 'Hmag', 'Kmag', 'W1mag', 'W2mag', 'W3mag', 'Y-J', 'J-H', 'H-K', 'J-K', 'W1-W2', 'W2-W3']].to_numpy().reshape(1, -1))[0]), 1)
                

                #Loading Neural Network

                model2 = torch.jit.load(models[-1])
                model2.eval()
                x = torch.tensor(dfcombined.iloc[i][['Ymag', 'Jmag', 'Hmag', 'Kmag', 'W1mag', 'W2mag', 'W3mag', 'Y-J', 'J-H', 'H-K', 'J-K', 'W1-W2', 'W2-W3']].astype("float").to_numpy(), dtype = torch.float)
                with torch.no_grad():
                    nnpredictor = model2(x)
                    nn_pred = round(float(nnpredictor[0].item()), 1)
                
                mindiff = min(abs(spt_pred_knn - spt_pred_tree), abs(spt_pred_knn - nn_pred), abs(spt_pred_tree - nn_pred))


                if abs(spt_pred_knn - spt_pred_tree) == mindiff:
                    sptpred = round((spt_pred_knn + spt_pred_tree)/2, 1)
                
                elif abs(spt_pred_knn - nn_pred) == mindiff:
                    sptpred = round((spt_pred_knn + nn_pred)/2, 1)
                
                elif abs(spt_pred_tree - nn_pred) == mindiff:
                    sptpred = round((spt_pred_tree + nn_pred)/2, 1)
                    
                if sptpred > 14:
                    sptpred = 14.0
              
                
                df.iloc[[i], df.columns.get_loc('sptype_classified')] = sptpred
                
                try:
                    tempdf = dfcombined[dfcombined['Sptfinal'] == sptpred]
                except IndexError:
                    tempdf = dfcombined[(sptpred - 0.5 <= dfcombined['Sptfinal'] ) & (dfcombined['Sptfinal'] <= sptpred + 0.5)]
                minch = 1000000
                bestA = 0
                a = 0
                

                for a in range(0, len(tempdf)):

                    ChiSquared = ((df.iloc[i]['Jmag'] - tempdf.iloc[a]['Jmag'])/Juncer)**2 + ((df.iloc[i]['Hmag'] - tempdf.iloc[a]['Hmag'])/Huncer)**2 + ((df.iloc[i]['Kmag'] - tempdf.iloc[a]['Kmag'])/Kuncer)**2 + ((df.iloc[i]['W1mag'] - tempdf.iloc[a]['W1mag'])/W1uncer)**2 + ((df.iloc[i]['W2mag'] - tempdf.iloc[a]['W2mag'])/W2uncer)**2
 
                    if ChiSquared <= minch:
                        minch = ChiSquared
                        bestA = a
                
                df.iloc[[i], df.columns.get_loc('Mass')] = tempdf.iloc[bestA]['Mass (Msun)']
                df.iloc[[i], df.columns.get_loc('Age')] = tempdf.iloc[bestA]['Age (Gyr)']
                df.iloc[[i], df.columns.get_loc('Temp')] = tempdf.iloc[bestA]['Temperature (Kelvin)']
                df.iloc[[i], df.columns.get_loc('logg')] = tempdf.iloc[bestA]['Gravity (logg)']
                df.iloc[[i], df.columns.get_loc('Metallicity')] = tempdf.iloc[bestA]['Msol']
            
                if sptpred >= 6.0 and df.iloc[i]['W1-W2'] <= 1.5:
                    df.iloc[[i], df.columns.get_loc('sptype_classified')] = None
            
             
                    
    return df.copy()
