import mne 
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt


def loadrandom(subjectpath, subject):

    random = []

    for file in os.listdir(subjectpath):
        if file.endswith('R.csv'):
                random.append(os.path.join(subjectpath, file))
                print(f'Random runs: {len(random)}')
    return random

def loadsequential(subjectpath, subject):
    sequential = []

    for file in os.listdir(subjectpath):
        if file.endswith('S.csv'):
                sequential.append(os.path.join(subjectpath, file))
                print(f'Sequential runs: {len(sequential)}')
    
    return sequential
            


def preprocess(eegdata, base, subjectpath, runfolder, highpass, lowpass):
    try:
            tmin = -2.0
            tmax = 5.0
  

            origin = os.listdir(runfolder)
            if 'epochs_epo.fif' in origin and 'cleaned_raw.fif' in origin and 'labels.npy' in origin:
                        print(f'[INFO] Preprocessed data already exists in {runfolder}. Skipping processing.')
                        raweeg = mne.io.read_raw_fif(os.path.join(runfolder, 'cleaned_raw.fif'), preload=True)
                        epochs = mne.read_epochs(os.path.join(runfolder, 'epochs_epo.fif'), preload=True)
                        adjustedlabels = np.load(os.path.join(runfolder, 'labels.npy'))
                        
                        print(f'[INFO] Loaded existing preprocessed data from {runfolder} with labels shape {adjustedlabels.shape}.')

                        return raweeg, epochs, adjustedlabels

            else:
                        eeg = os.path.join(subjectpath, eegdata)
                        df = pd.read_csv(eeg)

                        print(f'[INFO] Loaded file: {os.path.basename(eeg)}')

                        df['Marker'] = pd.to_numeric(df['Marker'], errors='coerce')
                        df['Marker'] = df['Marker'].ffill().fillna(0).astype(int)
                        print('Markers count:\n', df['Marker'].value_counts())


                        chnames = []

                        for ch in df.columns:
                            if ch not in ['Time', 'Marker']:
                                chnames.append(ch)
                        

                        ch_types = ['eeg'] * len(chnames)
                        num_channels = len(chnames)
                        sfreq = 512

                        raw_data = df[chnames].values.T

                        info = mne.create_info(chnames, sfreq, ch_types=ch_types)
                        raweeg = mne.io.RawArray(raw_data, info)

                        dropchannels = ['Ref1', 'Ref2']
                        raweeg.drop_channels(dropchannels)

                        raweeg.set_montage('standard_1020', on_missing='ignore')

                        print(f'[INFO] Filtering data with bandpass filter: {highpass} - {lowpass} Hz')
                        raweeg.filter(l_freq=highpass, h_freq=lowpass, fir_design='firwin')
                        raweeg.notch_filter(freqs=50.0, fir_design='firwin')

                        raweeg.set_eeg_reference(ref_channels='average')

                        epochmarker = df.index[(df['Marker'].shift(1)==2) & (df['Marker'].isin([3,4,5,6,7]))].tolist()
                        print(f'[INFO] Found {len(epochmarker)} epochs based on markers.')

                        presample = int(abs(tmin)*sfreq)
                        postsample = int(abs(tmax)*sfreq)

                        epochlength = presample + postsample

                        epochlist = []
                        validids = []

                        for idx in epochmarker:
                            if idx - presample >= 0 and idx + postsample < raweeg.n_times:
                                start, stop = idx - presample, idx + postsample
                                epochdata = raweeg[:, start:stop][0]
                                epochlist.append(epochdata)
                                validids.append(int(df.loc[idx, 'Marker']))

                            else:
                                print(f'[WARNING] Epoch at index {idx} is out of bounds and will be skipped.')

                            if not epochlist:
                                raise RuntimeError('[ERROR] No valid epochs were extracted from the data.')
                            
                        epocharray= np.array(epochlist)
                        events = np.array([[marker, 0, event] for marker, event in zip(epochmarker, validids)])

                        eventid = {str(event): event for event in np.unique(validids)}

                        print(f'[INFO] Creating Epochs object with {len(epocharray)} epochs.')

                        epochs = mne.EpochsArray(epocharray, raweeg.info, events=events, event_id=eventid, tmin=tmin)


                        keptepochs = [i for i, log in enumerate(epochs.drop_log) if len(log) == 0 ]
                        print(f'[INFO] Kept {len(keptepochs)} out of {len(epochs)} epochs after drop log filtering.')

                        adjustedlabels = np.array([event[2] for event in events])[keptepochs]
                        print(f'[INFO] Adjusted labels shape: {adjustedlabels.shape}')

                        print(f'[INFO] Saving data now.')

                        raweeg.save(os.path.join(runfolder, 'cleaned_raw.fif'), overwrite=True)
                        epochs.save(os.path.join(runfolder, 'epochs_epo.fif'), overwrite=True)
                        np.save(os.path.join(runfolder, 'labels.npy'), adjustedlabels)

            print(f'[INFO] Preprocessing complete. Returning labels and epochs. Labels shape: {adjustedlabels.shape}  ') 
            return raweeg, epochs, adjustedlabels

    except Exception as e:
        print(f"[ERROR] An error occurred during processing: {e}")
        return None, None, None

    
