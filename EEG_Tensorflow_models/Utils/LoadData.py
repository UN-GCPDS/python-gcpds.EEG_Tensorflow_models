from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing.preprocess import (exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.preprocessing.windowers import create_windows_from_events

import numpy as np
from scipy.signal import resample
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from mne.viz import plot_topomap


def load_dataset(dataset_name='BNCI2014001', subject_id=1, channels=None,
                 low_cut_freq_hz=4., high_cut_freq_hz=40.,
                 trial_start_offset_seconds=0.5, trial_stop_offset_seconds=0):
  
    """
    Load the data of a subject belonging to one of the available datasets in moabb package
    in a BaseConcatenate object of the braincode package.
    The default parameters load the data of the subject 1 of BCI Competition IV 2a database with the
    preprocessing applied in [Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018).
    EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. Journal of neural engineering, 15(5), 056013.]

    INPUT
    -----
    1. dataset_name (str): Name of the dataset in the moabb package. Default='BNCI2014001'
    2. subject_id (int+): subject to load. Default=1
    3. channels (list of str | None): name of channels to select, if none select all channels. Default=None
    4. low_cut_freq_hz (float+): High pass frequency. Default=4.
    5. high_cut_freq_hz (float+): low pass frequency. Default=40.
    6. trial_start_offset_seconds (float): start offset relative to trial onset. Default=0.5
    7. trial_stop_offset_seconds (float): stop offset relative to the end of motor imagey taks. Default=-1.5

    OUTPUT
    ------
    1. windows_dataset (BaseConcatenate Object): BaseConcatenate object of WindowDataset objects of braincode package.
    """
  
    dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=subject_id)

    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000
    
    if channels == None:
        preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False), # Keep EEG sensors
            Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
            Preprocessor('filter', l_freq=low_cut_freq_hz, h_freq=high_cut_freq_hz),  # Bandpass filter
            Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                        factor_new=factor_new, init_block_size=init_block_size)
        ]
    else:
        preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
            Preprocessor('pick_channels',ch_names=channels),
            Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
            Preprocessor('filter', l_freq=low_cut_freq_hz, h_freq=high_cut_freq_hz),  # Bandpass filter
            Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                        factor_new=factor_new, init_block_size=init_block_size)
        ]

    # Preprocess the data
    preprocess(dataset, preprocessors, n_jobs=-1)
    
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets]), 'The sample frequency has to be the same for all BaseDatasets'

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(trial_start_offset_seconds*sfreq),
        trial_stop_offset_samples=int(trial_stop_offset_seconds*sfreq),
        preload=True,
        n_jobs=-1
    )

    return windows_dataset


def get_data_from_BaseConcatDataset(BCDataset):
    """
    Get data in numpy arrays objects from BaseConcatenate object.
    INPUT
    -----
    1. BCDataset (BaseConcatenate object)
    OUTPUT
    ------
    1. X (4D array): shape (trials, channels, time_samples, 1) 
    2. y (1D arrat): shape (trials)
    """
    n_trials = len(BCDataset)
    X = np.zeros((n_trials,) + BCDataset[0][0].shape + (1,))
    y = np.zeros(n_trials)
    for trial in range(n_trials):
        X[trial, :, : , 0], y[trial], _ = BCDataset[trial]
    
    return X,y

def get_classes(X, y, classes_id):
  """
  Select specific classes from the Database.
  INPUT
  -----
  1. X (4D array): shape (trials, 1, channels, time_samples) 
  2. y (1D arrat): shape (trials)
  3. classes_id (list of int): id of classes to select.
  OUTPUT
  ------
  1. X (4D array): shape (trials_classes, 1, channels, time_samples) 
  2. y (1D arrat): shape (trials_classes)
  """
  bool_idx = np.zeros(y.shape[0], dtype=np.bool8)
  for cls_id in classes_id:
    bool_idx += (y == cls_id)

  return X[bool_idx], y[bool_idx]
    
