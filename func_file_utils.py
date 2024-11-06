# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:24:23 2024

@author: shahriar
"""

import os
import pickle


# example: save_data(data_dict,'sensitivity_matrices_hill.pkl')
def save_data(data, filename, folder='data'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, filename)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data successfully saved to {file_path}")
    
    
def load_data(filename, folder='data'):
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


# example: replace_data(['k_range'],[k_range],'sensitivity_matrices_hill.pkl')
def replace_data(keys, values, filename, folder='data'):
    if not isinstance(keys, list) or not isinstance(values, list):
        raise ValueError("keys and values must be lists.")
    if len(keys) != len(values):
        raise ValueError("keys and values lists must be of the same length.")
    data = load_data(filename, folder)
    for key, value in zip(keys, values):
        if key in data:
            data[key] = value
        else:
            raise KeyError(f"Key '{key}' not found in the data.")
    save_data(data, filename, folder)
    return data


# example: add_data_to_pkl(data_dict,'sensitivity_matrices_hill.pkl',)
def add_data_to_pkl(new_data, filename, folder='data'):
    file_path = os.path.join(folder, filename)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            existing_data = pickle.load(file)
    else:
        existing_data = {}
    existing_data.update(new_data)
    save_data(existing_data, filename, folder)

