from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing.preprocess import (exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.preprocessing.windowers import create_windows_from_events



def get_epochs(dset):
        y = []
        X = []
        for i in range(len(dset)):
            y.append(dset[i][1])
            X.append(np.expand_dims(dset[i][0],axis=[0,3]))
        
        y = np.asarray(y)
        X = np.concatenate(X,axis=0)
        return X,y



def load_dataset(dataset_name="BNCI2014001", subject_id=1, low_cut_hz = 4., high_cut_hz = 38., trial_start_offset_seconds = -0.5,
                 trial_stop_offset_seconds=0):

    dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=[subject_id])

    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                    factor_new=factor_new, init_block_size=init_block_size)
    ]

    # Transform the data
    preprocess(dataset, preprocessors)

    
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=int(trial_stop_offset_seconds*sfreq),
        preload=True,
    )

    splitted = windows_dataset.split('session')
    train_set = splitted['session_T']
    valid_set = splitted['session_E']

    X_train,y_train = get_epochs(train_set)
    X_valid,y_valid = get_epochs(valid_set)

    return X_train,y_train,X_valid,y_valid,sfreq