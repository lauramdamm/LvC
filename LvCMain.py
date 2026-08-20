import glob
import LvCFeatures as lvcf
import LvCPrePro as pmi
import LvCSubject as lvcs
import LvCGroup as lvcg
import LvCrossCondition as lvcc
import mne
import numpy as np
import os
import pandas as pd

SFREQ = 512

def process(data, base, subjectpath, orderpath, order, highpass, lowpass):
    epochsl = []
    labelsl = []
    paths = []

    for i, file in enumerate(data):
        runid = f'Run0{i+1}'
        runpath = os.path.join(orderpath, runid)
        os.makedirs(runpath, exist_ok=True)
        
        print(f'[INFO] Processing {file}')

        raw, epochs, labels = pmi.preprocess(eegdata = file, base = base, subjectpath = subjectpath, runfolder = runpath, lowpass = lowpass, highpass = highpass)
        epochsl.append(epochs)
        labelsl.append(labels)
        paths.append(runpath)

    allepochs = mne.concatenate_epochs(epochsl)
    alllabels = np.hstack(labelsl)

    return allepochs, alllabels, paths


def runsubject(base, sub, highpass, lowpass, feature):

    subj = f'Subject{int(sub):02d}'
    subjectpath = os.path.join(base, subj)
    os.makedirs(subjectpath, exist_ok=True)

    seqpath = os.path.join(subjectpath, 'Sequential')
    ranpath = os.path.join(subjectpath, 'Random')
    
    os.makedirs(seqpath, exist_ok=True)
    os.makedirs(ranpath, exist_ok=True)

    all_files = [f for f in os.listdir(subjectpath) if f.endswith('.csv')]

    print(f"For subject {sub}: {all_files}")
    
    seqdata = sorted([f for f in all_files if f.startswith(f'S{sub}R') and f.endswith('S.csv')])
    randata = sorted([f for f in all_files if f.startswith(f'S{sub}R') and f.endswith('R.csv')])

    print(f'[INFO] Found {len(seqdata)} Sequential and {len(randata)} Random files for {subj}.')

    if not seqdata and not randata:
        print(f'[WARN] No data found for {subj}. Skipping.')
        return None

    # Process data using the derived paths
    try:
        seqepochs, seqlabels, seqrunpath = process(seqdata, base, subjectpath, seqpath, 'Sequential', highpass, lowpass) if seqdata else (None, None)
        ranepochs, ranlabels, ranrunpath = process(randata, base, subjectpath, ranpath, 'Random', highpass, lowpass) if randata else (None, None)
    except Exception as e:
        print(f'[ERROR] Processing failed for {subj}: {e}')
        return None
    
    featurename = fchoice.upper()

    if featurename == 'CHAOS':
        feature = 'Chaos'
    elif featurename == 'LINEAR':
         feature = 'Linear'
    
    for typ in ('seq', 'ran'):
            if typ == 'seq':
                for path in seqrunpath:
                    outdir = os.path.join(path, feature)
                    os.makedirs(outdir, exist_ok=True)

                    epochs = seqepochs
                    labels = seqlabels

                    f_extraction = lvcf.FeatureExtraction.det_feat(epochs, highpass, lowpass, labels, outdir, feature, typ)
    
            elif typ == 'ran':
                for path in ranrunpath:
                    outdir = os.path.join(path, feature)
                    os.makedirs(outdir, exist_ok=True)

                    epochs = ranepochs
                    labels = ranlabels
                
                    f_extraction = lvcf.FeatureExtraction.det_feat(epochs, highpass, lowpass, labels, outdir, feature, typ)



if __name__ == '__main__':

    base = 'Dataset\sourcedata'


    fos = input('Feature Extraction (FE) or analysis (ST)? ').upper()


    if fos == 'FE':
        subjectstart= int(input('Which subject to start?: '))
        subjectend = int(input('Which subject to end?: '))
        highpass = int(input('Highpass Filter (i.e. 3): '))
        lowpass = int(input('Lowpass Filter (i.e. 45): '))
        fchoice = input('Linear or Chaos Feature: ')
        feature = fchoice.lower()

        for sub in range(subjectstart, subjectend):
            subid = f'{sub:02d}'
            
            print(f'\n[INFO] Scanning for Subject {subid}...')
            result = runsubject(base, subid, highpass, lowpass, feature)

    elif fos == 'ST':
         print('[INFO] Running extensive subjectwise statistical analysis.')
         lvcs.runitall()
         print('[INFO] Running extensive group-level and cross-condition statistical analysis.')
         lvcg.runitall()

    



                

