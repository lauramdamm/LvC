import numpy as np
import os
import pandas as pd
import random
import statsmodels.api as sm
from scipy import stats
import sys
import warnings



from statsmodels.stats.multitest import multipletests


from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from scipy.spatial.distance import mahalanobis


warnings.filterwarnings('ignore')
random.seed(42)


METRICS = ['HFD', 'Lyapunov_Exponent', 'Phase_Volume']
PAIRS = [('HFD', 'Lyapunov_Exponent'), ('HFD', 'Phase_Volume'), ('Lyapunov_Exponent', 'Phase_Volume')]
CONDITONS = ['Sequential', 'Random']
FREQUENCY_BANDS = [(3.5, 8), (8, 12), (12, 30), (30, 45)]



# ==============================================================================
# UTILITY FUNCTIONS (from Script 2 - enhanced)
# ==============================================================================

class Utils:
    """Statistical utilities shared across all analyses"""
    
    @staticmethod
    def fisher_z(r):
        """Safe Fisher Z transformation"""
        if r >= 1.0: r = 0.999
        if r <= -1.0: r = -0.999
        return 0.5 * np.log((1 + r) / (1 - r))
    
    @staticmethod
    def fisher_z_inverse(z):
        """Inverse Fisher Z"""
        return np.tanh(z)
    
    @staticmethod
    def hotelling_t2(group1, group2):
        """Hotelling's T² test for multivariate difference"""
        n1, n2 = len(group1), len(group2)
        if n1 < 4 or n2 < 4: 
            return None, None, None
        
        try:
            data1 = np.column_stack([group1[m] for m in METRICS])
            data2 = np.column_stack([group2[m] for m in METRICS])
            
            mean1 = np.mean(data1, axis=0)
            mean2 = np.mean(data2, axis=0)
            
            cov1 = np.cov(data1.T, ddof=1)
            cov2 = np.cov(data2.T, ddof=1)
            
            Sp = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)
            if np.linalg.cond(Sp) > 1e10:
                Sp += np.eye(3) * 1e-6
            
            Sp_inv = np.linalg.inv(Sp)
            diff = mean1 - mean2
            t2 = (n1 * n2 / (n1 + n2)) * np.dot(np.dot(diff, Sp_inv), diff)
            
            p = 3
            f_stat = ((n1 + n2 - p - 1) / (p * (n1 + n2 - 2))) * t2
            p_val = 1 - stats.f.cdf(f_stat, p, n1 + n2 - p - 1)
            md = np.sqrt(np.dot(np.dot(diff, Sp_inv), diff))
            
            return t2, p_val, md
        except np.linalg.LinAlgError:
            return None, None, None
    
    @staticmethod
    def extract_upper_triangle(corr_matrix):
        """Extract upper triangle correlations with NaN cleaning"""
        n = corr_matrix.shape[0]
        tri_idx = np.triu_indices(n, k=1)
        vec = corr_matrix.values[tri_idx]
        
        # Remove NaN pairs explicitly (Script 2 advantage)
        valid = ~np.isnan(vec)
        return vec[valid]
    
    @staticmethod
    def calculate_cohen_d(group1, group2):
        """Calculate Cohen's d effect size"""
        mean1, std1 = np.mean(group1), np.std(group1, ddof=1)
        mean2, std2 = np.mean(group2), np.std(group2, ddof=1)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
class OrganisationalCalls:
    def fisher_z(r):
        if r >= 1.0: r = 0.999
        if r <= -1.0: r = -0.999
        return 0.5 * np.log((1 + r) / (1 - r))


    def hotelling_t2(group1, group2):
        n1, n2 = len(group1), len(group2)
        if n1 < 4 or n2 < 4: return None, None, None
        
        # Stack data
        data1 = np.column_stack([group1[m] for m in ['HFD', 'Lyapunov_Exponent', 'Phase_Volume']])
        data2 = np.column_stack([group2[m] for m in ['HFD', 'Lyapunov_Exponent', 'Phase_Volume']])
        
        mean1 = np.mean(data1, axis=0)
        mean2 = np.mean(data2, axis=0)
        
        # Pooled Covariance Matrix
        cov1 = np.cov(data1.T, ddof=1)
        cov2 = np.cov(data2.T, ddof=1)
        
        # Handle singular covariance if variables are perfectly collinear
        try:
            Sp = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)
            # Add small regularisation if singular
            if np.linalg.cond(Sp) > 1e10:
                Sp += np.eye(3) * 1e-6
            
            Sp_inv = np.linalg.inv(Sp)
            
            diff = mean1 - mean2
            t2 = (n1 * n2 / (n1 + n2)) * np.dot(np.dot(diff, Sp_inv), diff)
            
            # F-statistic conversion
            p = 3
            f_stat = ((n1 + n2 - p - 1) / (p * (n1 + n2 - 2))) * t2
            p_val = 1 - stats.f.cdf(f_stat, p, n1 + n2 - p - 1)
            
            # Mahalanobis distance (effect size)
            md = np.sqrt(np.dot(np.dot(diff, Sp_inv), diff))
            
            return t2, p_val, md
        except np.linalg.LinAlgError:
            return None, None, None


    @staticmethod
    def loadlin(base_root):
        """Load Wintegrals.csv files with spectral band features."""
        print("\n[STEP 1a] Loading Linear Data...")
        dfs = []  # ← This should stay persistent across loop
        
        subjects = sorted([d for d in os.listdir(base_root) if d.startswith('Subject')])
        print(f"[INFO] Found {len(subjects)} subject folders")
        
        total_rows_before = 0
        total_rows_after = 0
        
        for subj in subjects:
            subj_path = os.path.join(base_root, subj)
            for condition in ['Sequential', 'Random']:
                cond_path = os.path.join(subj_path, condition)
                if not os.path.exists(cond_path): 
                    print(f"[SKIP] {subj}/{condition} - no path")
                    continue
                
                runs = sorted([d for d in os.listdir(cond_path) if d.startswith('Run0')])
                

                for run in runs:
                    file_path = os.path.join(cond_path, run, 'Linear', 'Wintegrals.csv')
                    if not os.path.exists(file_path): 
                        print(f'[DEBUG]{file_path}')
                    
                    df = pd.read_csv(file_path)
                    
                    total_rows_before += len(df)
                    
                    # Normalize column names
                    col_map = {c.lower(): c for c in df.columns}
                    rename_dict = {}
                    if 'channel' in col_map: rename_dict[col_map['channel']] = 'Channel'
                    if 'epoch' in col_map: rename_dict[col_map['epoch']] = 'Trial'
                    if 'label' in col_map: rename_dict[col_map['label']] = 'Class'
                    if 'frequency_band' in col_map: rename_dict[col_map['frequency_band']] = 'Frequency_Band'
                    
                    df.rename(columns=rename_dict, inplace=True)
                    
                    if 'Channel' not in df.columns: 
                        continue

                    
                    # ===== CRITICAL FIXES BELOW =====
                    df['Channel'] = df['Channel'].astype(str).str.upper().str.strip()
                    df['Class'] = df['Class'].astype(str).str.strip()  # ← Convert Class to STRING!
                    
                    df['Subject'] = subj
                    df['Condition'] = condition

                    # Check if any channels have missing frequency bands
                    freq_bands_per_channel = df.groupby('Channel')['Frequency_Band'].nunique()
                    missing_bands = freq_bands_per_channel[freq_bands_per_channel < len(FREQUENCY_BANDS)]
                    if len(missing_bands) > 0:
                        print(f"  [WARN] {len(missing_bands)} channels missing frequency bands:")
                        print(f"         {missing_bands.index.tolist()}")
                    
                    # Pivot to wide format
                    p_df = df.pivot_table(
                        index=['Subject', 'Condition', 'Trial', 'Class', 'Channel'],
                        columns='Frequency_Band',
                        values='integral', aggfunc='mean').reset_index()
                    
                    new_cols = {}
                    for col in p_df.columns:
                        if col not in ['Subject', 'Condition', 'Trial', 'Class', 'Channel']:
                            new_cols[col] = f"Lin_{col}"
                    p_df.rename(columns=new_cols, inplace=True)
                    
                    band_cols = [c for c in p_df.columns if c.startswith('Lin_')]
                    p_df[band_cols] = p_df[band_cols].fillna(0)
                    
                    total_rows_after += len(p_df)
                    dfs.append(p_df)  # ← Append to accumulating list!

        
        if not dfs:
            print("  [WARN] No linear data found.")
            return None
        
        df_linear = pd.concat(dfs, ignore_index=True)
        print(f"  [OK] Loaded {len(df_linear):,} linear trials from {total_rows_before:,} raw rows.")
        
        # Verify final structure
        print(f"  [DEBUG] Unique Subjects: {df_linear['Subject'].nunique()}")
        print(f"  [DEBUG] Unique Conditions: {df_linear['Condition'].unique()}")
        print(f"  [DEBUG] Unique Classes: {df_linear['Class'].unique()}")
        print(f"  [DEBUG] Class dtype: {df_linear['Class'].dtype}")
        
        return df_linear

        
    def loadcha(base_root):
        """Load Raw_Lyapunov_Full.csv files with IN-MEMORY Phase Volume log correction."""
        print("\n[STEP 1b] Loading Chaos Data (with automatic log-fix)...")
        dfs = []
        subjects = sorted([d for d in os.listdir(base_root) if d.startswith('Subject')])
        
        for subj in subjects:
            subj_path = os.path.join(base_root, subj)
            test_path = os.path.join(subj_path, 'FE')
            if not os.path.exists(test_path): continue
            
            for condition in ['Sequential', 'Random']:
                cond_path = os.path.join(test_path, condition, 'CC', 'Feature Analysis')
                if not os.path.exists(cond_path): continue
                
                file_path = os.path.join(cond_path, 'Raw_Lyapunov_Full.csv')
                if not os.path.exists(file_path): continue
                
                df = pd.read_csv(file_path)
                
                # ⭐ CRITICAL: Check and fix Phase Volume IN-MEMORY
                pv_max = df['Phase_Volume'].max()
                if pv_max > 1e6:
                    print(f"  [AUTO-FIX] Subject {subj}/{condition}: Phase Volume max={pv_max:.2e} → Applying log10")
                    df['Phase_Volume'] = np.where(
                        df['Phase_Volume'] > 0,
                        np.log10(df['Phase_Volume']),
                        0.0
                    )
                
                df['Subject'] = subj
                df['Condition'] = condition
                df['Trial'] = df['Trial'].astype(str)
                df['Class'] = df['Class'].astype(str).str.strip()  # ← Added .str.strip()
                
                if 'channel' in df.columns:
                    df.rename(columns={'channel': 'Channel'}, inplace=True)
                if 'Channel' in df.columns:
                    df['Channel'] = df['Channel'].astype(str).str.upper().str.strip()
                else:
                    continue
                
                required = ['Subject', 'Trial', 'Class', 'Channel', 'Condition', 
            'HFD', 'Lyapunov_Exponent', 'Phase_Volume',  # Existing
            'Phase_Centroid_X', 'Phase_Centroid_Y', 'Phase_Centroid_Z']  # ADD THESE


                if all(c in df.columns for c in required):
                    dfs.append(df[required])
        
        if not dfs:
            print("  [WARN] No chaos data found.")
            return None
        
        df_chaos = pd.concat(dfs, ignore_index=True)
        print(f"[INFO] Loaded {len(df_chaos)} chaos trials.")
        print(f"  Final Phase Volume range: [{df_chaos['Phase_Volume'].min():.2f}, {df_chaos['Phase_Volume'].max():.2f}]")
        return df_chaos

    def clean(df):
        """
        FIXED: Outlier removal using IQR OR Z-score (conservative union).
        Fixes Bug 1 from Script B where one overwrote the other.
        """
        if df.empty or 'Phase_Volume' not in df.columns:
            return df
        
        original_len = len(df)
        
        # Z-score method
        pv_clean = df['Phase_Volume'].dropna()
        z_scores = np.abs(stats.zscore(pv_clean))
        mask_z = pd.Series(False, index=df.index)
        valid_idx = pv_clean[z_scores < 3].index
        mask_z.loc[valid_idx] = True
        
        # IQR method
        q1 = df['Phase_Volume'].quantile(0.25)
        q3 = df['Phase_Volume'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask_iqr = (df['Phase_Volume'] >= lower_bound) | (df['Phase_Volume'] <= upper_bound)
        
        # UNION (conservative): keep if passes EITHER
        combined_mask = mask_z & mask_iqr
        
        df_clean = df[combined_mask].reset_index(drop=True)
        n_removed = original_len - len(df_clean)
        pct = (n_removed / original_len * 100) if original_len > 0 else 0
        
        print(f"  → Removed {n_removed} Phase_Volume outliers ({pct:.1f}%): IQR OR Z<3")
        return df_clean


    def harmonise(df_lin, df_cha):
        print("\n[INFO] Harmonising and Merging Datasets...")
        
        for df in [df_lin, df_cha]:
            if 'Channel' in df.columns:
                df['Channel'] = df['Channel'].astype(str).str.upper().str.strip()
        

        common_chs = set(df_lin['Channel']).intersection(set(df_cha['Channel']))
        if not common_chs:
            print("  [ERROR] No common channels between datasets.")
            return None
        df_lin = df_lin[df_lin['Channel'].isin(common_chs)].copy()
        df_cha = df_cha[df_cha['Channel'].isin(common_chs)].copy()
        
        df_cha = OrganisationalCalls.clean(df_cha)
        
        # Prepare base keys (grouping columns)
        base_keys = ['Subject', 'Trial', 'Channel', 'Condition']
        if 'Class' in df_lin.columns:
            base_keys.append('Class')
        if 'Class' in df_cha.columns:
            base_keys.append('Class')
        
        # Remove duplicates from base_keys just in case
        base_keys = list(dict.fromkeys(base_keys))
        
        # Identify feature columns to AGGREGATE (exclude ALL base_keys)
        feature_cols_lin = [c for c in df_lin.columns if c not in base_keys]
        
        # Double-check no base_key accidentally ended up in features
        feature_cols_lin = [c for c in feature_cols_lin if c not in base_keys]
        
        # Aggregate if duplicates exist
        if df_lin.duplicated(subset=base_keys).any():
            print(f"  → Aggregating {len(feature_cols_lin)} linear features across duplicates")
            agg_dict = {col: 'mean' for col in feature_cols_lin}
            final_lin = df_lin.groupby(base_keys)[list(agg_dict.keys())].agg(agg_dict).reset_index(drop=False)
            # Make sure only one 'Class' column survives
            if 'Class' in final_lin.columns:
                dupes = [col for col in final_lin.columns if col == 'Class']
                if len(dupes) > 1:
                    final_lin.drop(columns=dupes[1:], inplace=True)
        else:
            final_lin = df_lin.copy()
        
        # Explicitly remove any 'Class' duplicates
        if 'Class' in final_lin.columns:
            class_cols = [c for c in final_lin.columns if c.startswith('Class')]
            if len(class_cols) > 1:
                final_lin.drop(columns=class_cols[1:], inplace=True)
        
        # Verify 'Class' survived
        if 'Class' not in final_lin.columns and 'Class' in base_keys:
            print(f"  [WARNING] 'Class' missing from result. Available cols: {list(final_lin.columns)[:10]}")
        
        # Create merge key
        def create_key(row):
            parts = [
                str(row.get('Subject', '')),
                str(row.get('Condition', '')),
                str(row.get('Channel', '')),
                str(row.get('Trial', ''))
            ]
            if 'Class' in row and pd.notna(row.get('Class')):
                parts.append(str(row['Class']))
            return "|".join(parts)
        
        final_lin['Key'] = final_lin.apply(create_key, axis=1)
        df_cha['Key'] = df_cha.apply(create_key, axis=1)
        
        # Select chaos columns for merge
        cha_keep_cols = ['Key']
        if 'Class' in df_cha.columns:
            cha_keep_cols.append('Class')
        cha_features = [c for c in df_cha.columns if c in ['HFD', 'Lyapunov_Exponent', 'Phase_Volume']]
        cha_keep_cols.extend(cha_features)
        
        available_cha = [c for c in cha_keep_cols if c in df_cha.columns]
        df_cha_select = df_cha[available_cha].copy()
        
        merged = pd.merge(final_lin, df_cha_select, on='Key', how='inner')
        
        if len(merged) == 0:
            print("  [ERROR] Merge resulted in 0 rows.")
            return None
        
        id_fix = ['Subject', 'Trial', 'Channel', 'Condition', 'Class']
        for name in id_fix:
            x_col, y_col = f"{name}_x", f"{name}_y"
            if x_col in merged.columns and y_col in merged.columns:
                merged.rename(columns={x_col: name}, inplace=True)
                merged.drop(columns=[y_col], inplace=True)
        
        # Final verification
        required_final = ['Subject', 'Trial', 'Channel', 'Condition', 'Class']
        missing = [c for c in required_final if c not in merged.columns]
        if missing:
            print(f"  [FATAL] Missing critical columns: {missing}")
            print(f"  Available: {list(merged.columns)[:15]}")
            return None
        
        # Check for duplicates again
        for col in required_final:
            if col in merged.columns:
                dupe_count = sum([1 for c in merged.columns if c == col])
                if dupe_count > 1:
                    print(f"  [WARN] Column '{col}' appears {dupe_count} times!")
        
        print(f"  [INFO] Merged dataset: {len(merged)} trials.")
        print(f"    Linear Features: {[c for c in merged.columns if c.startswith('Lin_')][:5]}...")
        print(f"    Chaos Features: HFD, Lyapunov_Exponent, Phase_Volume")
        return merged

class subjectwise:
    @staticmethod
    def load(base, subj):
        subjp = os.path.join(base, subj)
        if not os.path.exists(subjp): return None
        data = []
        randp = os.path.join(subjp, 'Test', 'Random', 'HV', 'Feature Analysis')
        seqp = os.path.join(subjp, 'Test', 'Sequential', 'HV', 'Feature Analysis')
        required_cols = ['Trial', 'Class', 'Channel', 'Lyapunov_Exponent', 'HFD', 'Phase_Volume']
        
        def dictcheck(path, condition):
            csvp = os.path.join(path, 'Raw_Lyapunov_Full.csv')
            if not os.path.exists(csvp):
                return None
                
            df = pd.read_csv(csvp)
            if not all(c in df.columns for c in required_cols):
                return None
            '''
            # === APPLY CHANNEL SELECTION HERE ===
            df['Channel'] = df['Channel'].astype(str)
            valid_mask = df['Channel'].isin(TARGET_CHANNELS)
            n_removed = (~valid_mask).sum()
            df = df[valid_mask]
            
            print(f"      Removed {n_removed} rows (non-target channels)")
            '''
            if len(df) == 0:
                return None
            
            df['Condition'] = condition
            return df

        if os.path.exists(randp):
            df_rand = dictcheck(randp, 'Random')
            if df_rand is not None:
                data.append(df_rand)
                
        if os.path.exists(seqp):
            df_seq = dictcheck(seqp, 'Sequential')
            if df_seq is not None:
                data.append(df_seq)
                    
        if not data: return None
        
        combined = pd.concat(data, ignore_index=True)
        combined['Subject'] = subj
        combined['Class'] = combined['Class'].astype(str)
        combined['Condition'] = combined['Condition'].astype('category')
        return combined

    def calcova(df, subject_name):
        if df is None or len(df) == 0: return None
        metrics = ['HFD', 'Lyapunov_Exponent', 'Phase_Volume']
        unique_classes = sorted([str(c) for c in df['Class'].unique()])
        
        results = {
            'subject': subject_name,
            'finger_correlations': {'Random': {}, 'Sequential': {}},
            'finger_shifts': {'Random': {}, 'Sequential': {}},
            'channel_finger_diff': {'Random': {}, 'Sequential': {}},
            'multivariate': {'Random': {}, 'Sequential': {}},
            'raw_data_cache': {}
        }

        def process(cond_df, cond_label):
            if len(cond_df) == 0: return
            
            for cls in unique_classes:
                cls_df = cond_df[cond_df['Class'] == str(cls)]
                clean = cls_df[metrics].dropna()
                if len(clean) >= 10:
                    results['raw_data_cache'][f"{cond_label}_{cls}"] = clean.reset_index(drop=True)
            
            finger_r_data = {cls: {} for cls in unique_classes}
            pairs = [('HFD', 'Lyapunov_Exponent'), ('HFD', 'Phase_Volume'), ('Lyapunov_Exponent', 'Phase_Volume')]
            
            for cls in unique_classes:
                cls_df = cond_df[cond_df['Class'] == str(cls)]
                if len(cls_df) < 20: continue
                clean = cls_df[metrics].dropna()
                if len(clean) < 20: continue
                
                n_trials = len(clean)
                finger_r_data[cls]['n'] = n_trials
                finger_r_data[cls]['raw_corr'] = {}
                finger_r_data[cls]['fisher_z'] = {}
                finger_r_data[cls]['se'] = {}
                finger_r_data[cls]['partial_corr'] = {} 
                finger_r_data[cls]['multi_r2'] = {}     
                
                for m1, m2 in pairs:
                    x, y = clean[m1].values, clean[m2].values
                    if np.std(x) > 0 and np.std(y) > 0:
                        r_val = np.corrcoef(x, y)[0, 1]
                        if not np.isnan(r_val):
                            z_val = OrganisationalCalls.fisher_z(r_val)
                            se_val = 1.0 / np.sqrt(n_trials - 3)
                            pair_key = f"{m1}-{m2}"
                            finger_r_data[cls]['raw_corr'][pair_key] = float(r_val)
                            finger_r_data[cls]['fisher_z'][pair_key] = float(z_val)
                            finger_r_data[cls]['se'][pair_key] = float(se_val)

                # ---------------------------------------------------------
                # 2. PARTIAL CORRELATIONS (THE MISSING CODE)
                # ---------------------------------------------------------
                r_hfd_lya = clean['HFD'].corr(clean['Lyapunov_Exponent'])
                r_hfd_phase = clean['HFD'].corr(clean['Phase_Volume'])
                r_lya_phase = clean['Lyapunov_Exponent'].corr(clean['Phase_Volume'])
                
                if any(np.isnan([r_hfd_lya, r_hfd_phase, r_lya_phase])):
                    p_vals = {k: 0.0 for k in ['HFD-Lyapunov_Exponent|Phase_Volume', 
                                             'HFD-Phase_Volume|Lyapunov_Exponent', 
                                             'Lyapunov_Exponent-Phase_Volume|HFD']}
                else:
                    # Partial HFD-Lya | Phase
                    num = r_hfd_lya - (r_hfd_phase * r_lya_phase)
                    den = np.sqrt((1-r_hfd_phase**2)*(1-r_lya_phase**2))
                    r_partial_1 = num/den if abs(den) > 1e-6 else 0.0
                    
                    # Partial HFD-Phase | Lya
                    num2 = r_hfd_phase - (r_hfd_lya * r_lya_phase)
                    den2 = np.sqrt((1-r_hfd_lya**2)*(1-r_lya_phase**2))
                    r_partial_2 = num2/den2 if abs(den2) > 1e-6 else 0.0

                    # Partial Lya-Phase | HFD
                    num3 = r_lya_phase - (r_hfd_lya * r_hfd_phase)
                    den3 = np.sqrt((1-r_hfd_lya**2)*(1-r_hfd_phase**2))
                    r_partial_3 = num3/den3 if abs(den3) > 1e-6 else 0.0
                    
                    p_vals = {
                        'HFD-Lyapunov_Exponent|Phase_Volume': float(r_partial_1),
                        'HFD-Phase_Volume|Lyapunov_Exponent': float(r_partial_2),
                        'Lyapunov_Exponent-Phase_Volume|HFD': float(r_partial_3)
                    }
                finger_r_data[cls]['partial_corr'] = p_vals

                # ---------------------------------------------------------
                # 3. MULTIPLE R-SQUARED (THE MISSING CODE)
                # ---------------------------------------------------------
                r_sq_vals = {}
                try:
                    # Predict Phase from HFD+Lya
                    y = clean['Phase_Volume'].values
                    X = sm.add_constant(clean[['HFD', 'Lyapunov_Exponent']].values)
                    model = sm.OLS(y, X).fit()
                    r_sq_vals['Phase_Predicted_by_HFD_Lya'] = float(model.rsquared)
                except: r_sq_vals['Phase_Predicted_by_HFD_Lya'] = 0.0

                try:
                    # Predict HFD from Lya+Phase
                    y = clean['HFD'].values
                    X = sm.add_constant(clean[['Lyapunov_Exponent', 'Phase_Volume']].values)
                    model = sm.OLS(y, X).fit()
                    r_sq_vals['HFD_Predicted_by_Lya_Phase'] = float(model.rsquared)
                except: r_sq_vals['HFD_Predicted_by_Lya_Phase'] = 0.0

                try:
                    # Predict Lya from HFD+Phase
                    y = clean['Lyapunov_Exponent'].values
                    X = sm.add_constant(clean[['HFD', 'Phase_Volume']].values)
                    model = sm.OLS(y, X).fit()
                    r_sq_vals['Lya_Predicted_by_HFD_Phase'] = float(model.rsquared)
                except: r_sq_vals['Lya_Predicted_by_HFD_Phase'] = 0.0
                
                finger_r_data[cls]['multi_r2'] = r_sq_vals

            results['finger_correlations'][cond_label] = finger_r_data
            

            # === ADD THIS: FINGER SHIFTS ===
            # Compare Fisher-Z values between fingers for each metric pair
            finger_classes = sorted([c for c in finger_r_data.keys() if 'fisher_z' in finger_r_data[c]])
            pairs_list = [('HFD', 'Lyapunov_Exponent'), ('HFD', 'Phase_Volume'), ('Lyapunov_Exponent', 'Phase_Volume')]

            for i in range(len(finger_classes)):
                for j in range(i + 1, len(finger_classes)):
                    f_i = finger_classes[i]
                    f_j = finger_classes[j]
                    comp_key = f"{f_i}_vs_{f_j}"

                    if comp_key not in results['finger_shifts'][cond_label]:
                        results['finger_shifts'][cond_label][comp_key] = {}

                    for m1, m2 in pairs_list:
                        pair_key = f"{m1}-{m2}"

                        z_i = finger_r_data[f_i].get('fisher_z', {}).get(pair_key)
                        z_j = finger_r_data[f_j].get('fisher_z', {}).get(pair_key)
                        se_i = finger_r_data[f_i].get('se', {}).get(pair_key)
                        se_j = finger_r_data[f_j].get('se', {}).get(pair_key)

                        if z_i is None or z_j is None or se_i is None or se_j is None:
                            continue

                        diff_z = z_i - z_j
                        # Z-test for difference between two independent correlations
                        se_diff = np.sqrt(se_i**2 + se_j**2)
                        if se_diff > 0:
                            z_stat = diff_z / se_diff
                            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        else:
                            z_stat = 0
                            p_val = 1.0

                        r_i = finger_r_data[f_i].get('raw_corr', {}).get(pair_key, 0)
                        r_j = finger_r_data[f_j].get('raw_corr', {}).get(pair_key, 0)

                        results['finger_shifts'][cond_label][comp_key][pair_key] = {
                            'diff_z': float(diff_z),
                            'p_value': float(p_val),
                            'r_i': float(r_i),
                            'r_j': float(r_j)
                        }

            # === ADD THIS: CHANNEL-LEVEL FINGER DIFFERENCES ===
            # Per-channel correlations, compared across fingers
            if 'Channel' in cond_df.columns:
                channels_present = cond_df['Channel'].unique()

                for ch in channels_present:
                    ch_df = cond_df[cond_df['Channel'] == ch]
                    

                    if ch not in results['channel_finger_diff'][cond_label]:
                        results['channel_finger_diff'][cond_label][ch] = {}

                    # Compute per-finger correlations for this channel
                    ch_finger_corr = {}
                    for cls in unique_classes:
                        cls_ch_df = ch_df[ch_df['Class'] == str(cls)]
                        clean_ch = cls_ch_df[metrics].dropna()
                        if len(clean_ch) < 20:
                            continue

                        ch_finger_corr[cls] = {}
                        for m1, m2 in pairs_list:
                            if np.std(clean_ch[m1]) > 0 and np.std(clean_ch[m2]) > 0:
                                r_val = clean_ch[m1].corr(clean_ch[m2])
                                if not np.isnan(r_val):
                                    ch_finger_corr[cls][f"{m1}-{m2}"] = r_val

                    # Compare finger pairs within this channel
                    finger_keys = sorted(ch_finger_corr.keys())
                    for i in range(len(finger_keys)):
                        for j in range(i + 1, len(finger_keys)):
                            f_i = finger_keys[i]
                            f_j = finger_keys[j]
                            comp_key = f"{f_i}_vs_{f_j}"

                            for m1, m2 in pairs_list:
                                pair_key = f"{m1}-{m2}"

                                if pair_key not in ch_finger_corr[f_i] or pair_key not in ch_finger_corr[f_j]:
                                    continue

                                r_i = ch_finger_corr[f_i][pair_key]
                                r_j = ch_finger_corr[f_j][pair_key]

                                z_i = OrganisationalCalls.fisher_z(r_i)
                                z_j = OrganisationalCalls.fisher_z(r_j)

                                n_i = len(ch_df[ch_df['Class'] == str(f_i)][metrics].dropna())
                                n_j = len(ch_df[ch_df['Class'] == str(f_j)][metrics].dropna())

                                se_diff = np.sqrt(1/(n_i-3) + 1/(n_j-3))
                                diff_z = z_i - z_j

                                if se_diff > 0:
                                    z_stat = diff_z / se_diff
                                    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                                else:
                                    z_stat = 0
                                    p_val = 1.0

                                winner = f_i if diff_z > 0 else f_j
                                direction = f"{f_i}>{f_j}" if diff_z > 0 else f"{f_j}>{f_i}"
                                significant = 'Yes' if p_val < 0.05 else 'No'

                                if pair_key not in results['channel_finger_diff'][cond_label][ch]:
                                    results['channel_finger_diff'][cond_label][ch][pair_key] = {}

                                results['channel_finger_diff'][cond_label][ch][pair_key][comp_key] = {
                                    'winner': winner,
                                    'direction': direction,
                                    'mean_diff': float(diff_z),
                                    'abs_diff': float(abs(diff_z)),
                                    'p_value': float(p_val),
                                    'z_stat': float(z_stat),
                                    'significant': significant,
                                    'r_i': float(r_i),
                                    'r_j': float(r_j)
                                }

            # CRITICAL: Populate multivariate dict for downstream access
            results['multivariate'][cond_label] = {}
            for cls in unique_classes:
                if cls in finger_r_data:
                    results['multivariate'][cond_label][cls] = {
                        'partial': finger_r_data[cls].get('partial_corr', {}),
                        'multi_r2': finger_r_data[cls].get('multi_r2', {}),
                        'n': finger_r_data[cls].get('n', 0)
        }
            

        for cond in ['Random', 'Sequential']:
            cond_df = df[df['Condition'] == cond]
            process(cond_df, cond)
            
        return results

    def save(results, output_dir):
        subj = results['subject']
        subj_dir = os.path.join(output_dir, subj)
        os.makedirs(subj_dir, exist_ok=True)
        
        saved_files = []
        
        # === CSV 1: Finger Correlations ===
        rows = []
        metrics_pairs = [('HFD', 'Lyapunov_Exponent'), ('HFD', 'Phase_Volume'), ('Lyapunov_Exponent', 'Phase_Volume')]
        
        for cond_label in ['Random', 'Sequential']:
            finger_data = results['finger_correlations'].get(cond_label, {})
            multiv_data = results['multivariate'].get(cond_label, {})
            
            for cls, data in finger_data.items():
                raw_corr = data.get('raw_corr', {})
                fisher_z_vals = data.get('fisher_z', {})
                se_vals = data.get('se', {})
                partial_corr = data.get('partial_corr', {})
                multi_r2 = data.get('multi_r2', {})
                n_trials = data.get('n', 0)
                
                for pair, r_val in raw_corr.items():
                    z_val = fisher_z_vals.get(pair)
                    se = se_vals.get(pair)
                    p_val = None
                    is_sig_zero = False
                    ci_low = None
                    ci_high = None
                    
                    if n_trials >= 10 and se and se > 0 and z_val is not None:
                        z_stat = abs(z_val) / se
                        p_val = 2 * (1 - stats.norm.cdf(z_stat))
                        is_sig_zero = p_val < 0.05
                        ci_low = z_val - 1.96 * se
                        ci_high = z_val + 1.96 * se
                    
                    rows.append({
                        'Subject': subj,
                        'Condition': cond_label,
                        'Finger_Class': cls,
                        'Type': 'Pearson_Correlation',
                        'Metric_Pair': pair,
                        'Pearson_R': round(r_val, 4),
                        'Fisher_Z': round(z_val, 4) if z_val is not None else None,
                        'N_Trials': n_trials,
                        'SE_Z': round(se, 4) if se is not None else None,
                        'CI_95_Low': round(ci_low, 4) if ci_low is not None else None,
                        'CI_95_High': round(ci_high, 4) if ci_high is not None else None,
                        'Partial_Corr': None,
                        'Multi_R2': None
                    })

                for key, p_val in partial_corr.items():
                    if n_trials >= 10:
                        r_part = p_val
                        t_stat = r_part * np.sqrt((n_trials - 3) / (1 - r_part**2 + 1e-6))
                        p_sig = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_trials-3))
                        sig_str = 'Yes' if p_sig < 0.05 else 'No'
                    else:
                        sig_str = 'No'
                        p_sig = 1.0

                    rows.append({
                        'Subject': subj,
                        'Condition': cond_label,
                        'Finger_Class': cls,
                        'Type': 'Partial_Correlation',
                        'Metric_Pair': key, # e.g., "HFD-Lya|Phase"
                        'Pearson_R': round(p_val, 4),
                        'Fisher_Z': None,
                        'N_Trials': n_trials,
                        'SE_Z': None,
                        'CI_95_Low': None,
                        'CI_95_High': None,
                        'Partial_Corr': round(p_val, 4),
                        'Multi_R2': None
                    })

                for key, r2_val in multi_r2.items():
                    rows.append({
                        'Subject': subj,
                        'Condition': cond_label,
                        'Finger_Class': cls,
                        'Type': 'Multiple_R2',
                        'Metric_Pair': key, 
                        'Pearson_R': None,
                        'Fisher_Z': None,
                        'N_Trials': n_trials,
                        'SE_Z': None,
                        'CI_95_Low': None,
                        'CI_95_High': None,
                        'Partial_Corr': None,
                        'Multi_R2': round(r2_val, 4)
                    })
        
        if rows:
            df = pd.DataFrame(rows)
            path = os.path.join(subj_dir, 'Finger_Correlations.csv')
            df.to_csv(path, index=False)
            saved_files.append(path)
        
        # === CSV 2: Finger Shifts ===
        rows = []
        for cond_label in ['Random', 'Sequential']:
            shift_data = results['finger_shifts'].get(cond_label, {})
            for comp_key, pairs_dict in shift_data.items():
                for pair, pair_stats in pairs_dict.items():  
                    diff_z = pair_stats.get('diff_z', 0)
                    p_val = pair_stats.get('p_value', 1.0)
                    r_i = pair_stats.get('r_i', 0)
                    r_j = pair_stats.get('r_j', 0)
                    
                    rows.append({
                        'Subject': subj,
                        'Condition': cond_label,
                        'Comparison': comp_key,
                        'Metric_Pair': pair,
                        'Diff_Z_Score': round(diff_z, 4),
                        'R_Finger_A': round(r_i, 4),
                        'R_Finger_B': round(r_j, 4),
                        'Direction': 'A>B' if diff_z > 0 else 'B>A',
                        'P_Value': round(p_val, 8),
                        'Is_Significant': 'Yes' if p_val < 0.05 else 'No',
                    })
        
        if rows:
            df = pd.DataFrame(rows)
            path = os.path.join(subj_dir, 'Finger_Shifts.csv')
            df.to_csv(path, index=False)
            saved_files.append(path)
        
        # === CSV 3: Channel-Level Differences ===
        rows = []
        for cond_label in ['Random', 'Sequential']:
            ch_data = results['channel_finger_diff'].get(cond_label, {})
            for chan, pairs_dict in ch_data.items():
                for pair, pair_metrics_dict in pairs_dict.items():  
                    for comp_key, result_data in pair_metrics_dict.items(): 
                        rows.append({
                            'Subject': subj,
                            'Condition': cond_label,
                            'Channel': chan,
                            'Metric_Pair': pair,
                            'Finger_Comparison': comp_key,
                            'Winner_Finger': result_data.get('winner', ''),
                            'Direction': result_data.get('direction', ''),
                            'Mean_Diff_Z': round(result_data.get('mean_diff', 0), 4),
                            'Abs_Mean_Diff': round(result_data.get('abs_diff', 0), 4),
                            'P_Value': round(result_data.get('p_value', 1.0), 8),
                            'Z_Statistic': round(result_data.get('z_stat', 0), 4),
                            'Is_Significant': result_data.get('significant', 'No'),
                            'Corr_Finger_A': round(result_data.get('r_i', 0), 4),
                            'Corr_Finger_B': round(result_data.get('r_j', 0), 4)
                        })
        
        if rows:
            df = pd.DataFrame(rows)
            path = os.path.join(subj_dir, 'Channel_Level_Differences.csv')
            df.to_csv(path, index=False)
            saved_files.append(path)
        
        # === CSV 4: Summary Table (FIXED LOGIC) ===
        summary_rows = []
        for cond_label in ['Random', 'Sequential']:
            n_channels = len(results['channel_finger_diff'].get(cond_label, {}))
            n_shifts = len(results['finger_shifts'].get(cond_label, {}))
            
            # FIX: Count based on the actual P_VALUE in the data structure, not just the flag string
            n_sig_shifts = 0
            shift_data = results['finger_shifts'].get(cond_label, {})
            for comp_key, pairs_dict in shift_data.items():
                for pair, pair_stats in pairs_dict.items(): 
                    if pair_stats.get('p_value', 1.0) < 0.05:
                        n_sig_shifts += 1
            
            n_sigs = 0
            ch_data = results['channel_finger_diff'].get(cond_label, {})
            for chan, pairs_dict in ch_data.items():
                for pair, pair_metrics_dict in pairs_dict.items(): 
                    for comp_key, result_data in pair_metrics_dict.items(): 
                        if result_data.get('p_value', 1.0) < 0.05:
                            n_sigs += 1
            
            summary_rows.append({
                'Subject': subj,
                'Condition': cond_label,
                'N_Channels_Analysed': n_channels,
                'N_Finger_Comparisons': n_shifts,
                'N_Significant_Shifts_Global': n_sig_shifts,
                'N_Significant_Channel_Diffs': n_sigs
            })
        
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            path = os.path.join(subj_dir, 'Summary.csv')
            df.to_csv(path, index=False)
            saved_files.append(path)
        
        return saved_files 

class MultivariateFingerDiffAnalysis:
    """
    [INSERT IN PAPER] To consider if a 3 dimensional multivariate chaos feature space manages finger differentiability, Hotelling's T was computed in a pairwise manner. Furthermore, any significant pairing had their effect size assessed via Mahalanobis Distance which is the multivariate counterpart to Cohen's D.
    """
    def __init__(self, subject_results):
        self.results = subject_results
    
    def runana(self, output_dir):
        print("\n" + "="*70)
        print("[INFO] 3D MULTIVARIATE FINGER DIFFERENTIATION")
        print("="*70)
        print("Method: Hotelling's T² (Pairwise) & Mahalanobis Distance")
        print("-"*70)

        analyse_dir = os.path.join(output_dir, 'Multivariate_Finger_Diff')
        os.makedirs(analyse_dir, exist_ok=True)

        metrics = ['HFD', 'Lyapunov_Exponent', 'Phase_Volume']
        conditions = ['Random', 'Sequential']
        all_pairs_data = []

        total_subjects = len(self.results)
        cached_keys = []
        
        print(f"\n[INFO] Checking data cache across {total_subjects} subjects...")
        for subj_res in self.results:
            cache = subj_res.get('raw_data_cache', {})
            if cache:
                cached_keys.extend([k for k in cache.keys()])
        
        print(f"[INFO] Total cached key combinations found: {len(cached_keys)}")
        if cached_keys:
            print(f"Sample keys: {cached_keys[:6]}")
        else:
            print("[ERROR] No raw data cache found! Ensure calcova populates 'raw_data_cache'")

        # === MAIN ANALYSIS LOOP ===
        for subj_idx, subj_res in enumerate(self.results):
            subj_id = subj_res['subject']
            cache = subj_res.get('raw_data_cache', {})
            
            if not cache:
                print(f"[SKIP] Subject {subj_id}: No raw_data_cache")
                continue
                
            for cond in conditions:
                fingers_in_cond = sorted(list(set([
                    k.split('_')[1] for k in cache.keys() 
                    if k.startswith(f"{cond}_")
                ])))
                
                if len(fingers_in_cond) < 2:
                    print(f"  [SKIP] Subject {subj_id} | Condition: {cond} | Only {len(fingers_in_cond)} finger(s)")
                    continue
                
                print(f"\n[SUBJ {subj_id}] Condition: {cond} | Fingers: {fingers_in_cond}")
                
                comparisons = []
                for i in range(len(fingers_in_cond)):
                    for j in range(i + 1, len(fingers_in_cond)):
                        f_i = fingers_in_cond[i]
                        f_j = fingers_in_cond[j]
                        
                        key_i = f"{cond}_{f_i}"
                        key_j = f"{cond}_{f_j}"
                        
                        if key_i not in cache or key_j not in cache:
                            print(f"    Missing keys: {key_i} or {key_j} in cache")
                            continue
                        
                        data_i = cache[key_i][metrics]
                        data_j = cache[key_j][metrics]
                        
                        t2, p_val, md = OrganisationalCalls.hotelling_t2(data_i, data_j)
                        
                        if t2 is not None:
                            sig = "NA"
                            comparisons.append({
                                'Subject': subj_id,
                                'Condition': cond,
                                'Finger_A': f_i,
                                'Finger_B': f_j,
                                'T2_Statistic': round(t2, 4),
                                'P_Value_Raw': round(p_val, 6), 
                                'P_Value_FDR': None,               
                                'Is_Significant_Raw': p_val < 0.05,  
                                'Mahalanobis_Distance': round(md, 4),
                                'Interpretation': "Pending FDR Correction"
                            })
                        else:
                            print(f"    Hotelling failed for {f_i} vs {f_j} (covariance singularity?)")
                
                if comparisons:
                    df_comp = pd.DataFrame(comparisons)
                    path = os.path.join(analyse_dir, f"{subj_id}_{cond}_Pairs.csv")
                    df_comp.to_csv(path, index=False)
                    all_pairs_data.extend(comparisons)
                    print(f"  → Saved {len(comparisons)} pairwise comparisons")

        # ====== APPLY BENJAMINI-HOCHBERG FDR CORRECTION ======
        if all_pairs_data:
            df_agg = pd.DataFrame(all_pairs_data)
            
            # Apply FDR separately per condition
            for cond in conditions:
                cond_indices = df_agg[df_agg['Condition'] == cond].index.tolist()
                n_tests = len(cond_indices)
                
                if n_tests == 0:
                    continue

                raw_pvals = df_agg.loc[cond_indices, 'P_Value_Raw'].values
                
                reject_fdr, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
                
                df_agg.loc[cond_indices, 'P_Value_FDR'] = pvals_corrected
                df_agg.loc[cond_indices, 'Is_Significant_FDR'] = reject_fdr.astype(str)  
                
                print(f"\n[FDR CORRECTED] Condition: {cond}")
                print(f"  Tests: {n_tests}")
                print(f"  Significant (uncorrected): {(raw_pvals < 0.05).sum()}")
                print(f"  Significant (FDR-corrected): {reject_fdr.sum()}")
            
            # Save aggregated results with FDR corrections
            df_agg_sorted = df_agg.sort_values(['Condition', 'Subject', 'Finger_A', 'Finger_B'])
            path_sum = os.path.join(analyse_dir, 'Group_Multivariate_Summary_FDR_Pairwise.csv')
            df_agg_sorted.to_csv(path_sum, index=False)
            
            print(f"\n[OK] Aggregated results saved to {path_sum}")
            
            # ====== COMPUTE GROUP-LEVEL SUMMARY ======
            summary_rows = []
            for cond in conditions:
                sub_df = df_agg_sorted[df_agg_sorted['Condition'] == cond]
                if len(sub_df) == 0: continue
                
                total_tests = len(sub_df)
                sig_tests_raw = sub_df[sub_df['Is_Significant_Raw'] == True]['T2_Statistic'].count()
                sig_tests_fdr = sub_df[sub_df['Is_Significant_FDR'] == 'True']['T2_Statistic'].count()
                avg_md = sub_df['Mahalanobis_Distance'].mean()
                
                # Finger-level counts (post-FDR)
                finger_sig_counts = {}
                for _, row in sub_df.iterrows():
                    if row['Is_Significant_FDR'] == 'True':
                        f_a = row['Finger_A']
                        f_b = row['Finger_B']
                        finger_sig_counts[f_a] = finger_sig_counts.get(f_a, 0) + 1
                        finger_sig_counts[f_b] = finger_sig_counts.get(f_b, 0) + 1
                
                summary_rows.append({
                    'Condition': cond,
                    'Total_Pairwise_Tests': total_tests,
                    'Significant_Unadjusted': int(sig_tests_raw),
                    'Significant_FDR_Corrected': int(sig_tests_fdr),
                    'Proportion_Significant_Unadj': round(sig_tests_raw / total_tests, 3),
                    'Proportion_Significant_FDR': round(sig_tests_fdr / total_tests, 3),
                    'Avg_Mahalanobis_Dist': round(avg_md, 4),
                    'Most_Distinct_Fingers_FDR': dict(sorted(finger_sig_counts.items(), key=lambda x: x[1], reverse=True)[:3]) if finger_sig_counts else {}
                })
            
            df_sum = pd.DataFrame(summary_rows)
            path_sum_summary = os.path.join(analyse_dir, 'Group_Multivariate_Summary_FDR.csv')
            df_sum.to_csv(path_sum_summary, index=False)
            

            print(f"\n[INFO] Saved FDR-corrected summary to {path_sum_summary}")
        else:
            print("\n[ERROR] No multivariate comparisons were successful.")

        print("\n[DONE] 3D Differentiation Analysis Complete.")
        return analyse_dir

class GroupCorrelationAnalysis:
    def __init__(self, subject_results):
        self.results = subject_results
        
    def aggz(self, conditions, pairs):
        aggregated = {}
        
        for cond in conditions:
            aggregated[cond] = {}
            for pair in pairs:
                aggregated[cond][pair] = {}
                
                all_fingers = set()
                for subj_res in self.results:
                    finger_data = subj_res.get('finger_correlations', {}).get(cond, {})
                    for f_class, f_data in finger_data.items():
                        if pair in f_data.get('fisher_z', {}):
                            all_fingers.add(f_class)
                            
                for f_class in all_fingers:
                    z_values = []
                    for subj_res in self.results:
                        finger_data = subj_res.get('finger_correlations', {}).get(cond, {})
                        if f_class in finger_data:
                            z_val = finger_data[f_class].get('fisher_z', {}).get(pair)
                            se_val = finger_data[f_class].get('se', {}).get(pair)
                            
                            if z_val is not None and se_val is not None and se_val > 0:
                                z_values.append(z_val)
                    
                    if len(z_values) >= 5:
                        aggregated[cond][pair][f_class] = z_values
                        
        return aggregated

    def runana(self, output_dir):
        metrics = ['HFD', 'Lyapunov_Exponent', 'Phase_Volume']
        pairs = [('HFD', 'Lyapunov_Exponent'), ('HFD', 'Phase_Volume'), ('Lyapunov_Exponent', 'Phase_Volume')]
        pair_names = [f"{p[0]}-{p[1]}" for p in pairs]
        conditions = ['Random', 'Sequential']
        
        print("\n[INFO] Running Group-Level Correlation Analysis...")

        agg_data = self.aggz(conditions, pair_names)
        
        if not any(agg_data[c] for c in conditions):
            print("[WARN] No sufficient data found for group analysis.")
            return None
        
        group_dir = os.path.join(output_dir, 'Group_Correlation_Analysis')
        os.makedirs(group_dir, exist_ok=True)

        df_pairwise = []

        for cond in conditions:
            for pair in pair_names:
                fingers = sorted(agg_data[cond][pair].keys())
                if len(fingers) < 2:
                    continue
                    
                print(f"\n[INFO] Condition: {cond}, Pair: {pair}")
                print(f"  Fingers analysed: {len(fingers)}")

                for f in fingers:
                    z_vals = agg_data[cond][pair][f]
                    n_subj = len(z_vals)
                    mean_z = np.mean(z_vals)
                    std_z = np.std(z_vals, ddof=1)
                    sem_z = std_z / np.sqrt(n_subj)
                    
                    t_stat, p_val_t = stats.ttest_1samp(z_vals, 0.0)
                    w_stat, p_val_w = stats.wilcoxon(z_vals)
                    
                    mean_r = np.tanh(mean_z)
                    

                finger_pairs = [(fingers[i], fingers[j]) for i in range(len(fingers)) for j in range(i + 1, len(fingers))]
                
                for f_i, f_j in finger_pairs:
                    z_i = agg_data[cond][pair][f_i]
                    z_j = agg_data[cond][pair][f_j]
                    
                    if len(z_i) != len(z_j):
                        min_len = min(len(z_i), len(z_j))
                        z_i = z_i[:min_len]
                        z_j = z_j[:min_len]
                        
                    diffs = np.array(z_i) - np.array(z_j)
                    mean_diff = np.mean(diffs)
                    std_diff = np.std(diffs, ddof=1)
                    
                    t_stat, p_val_t = stats.ttest_rel(z_i, z_j)
                    w_stat, p_val_w = stats.wilcoxon(z_i, z_j)
                    
                    direction = f"{f_i}>{f_j}" if mean_diff > 0 else f"{f_j}>{f_i}"
                    
                    df_pairwise.append({
                        'Condition': cond,
                        'Metric_Pair': pair,
                        'Finger_A': f_i,
                        'Finger_B': f_j,
                        'Comparison': f"F{f_i}_vs_F{f_j}",
                        'Mean_Diff_Z': round(mean_diff, 4),
                        'Std_Diff': round(std_diff, 4),
                        'T_Statistic': round(t_stat, 4),
                        'P_Value_T_Test': round(p_val_t, 6),
                        'P_Value_Wilcoxon': round(p_val_w, 6),
                        'Direction': direction,
                        'Significant_T': 'Yes' if p_val_t < 0.05 else 'No',
                        'Significant_Wilcoxon': 'Yes' if p_val_w < 0.05 else 'No'
                    })

        if df_pairwise:
            df_out = pd.DataFrame(df_pairwise)
            path = os.path.join(group_dir, 'Group_Pairwise_Finger_Comparisons.csv')
            df_out.to_csv(path, index=False)
            print(f"[OK] Saved Group Pairwise Comparisons: {path}")

        print("\n[DONE] Group Correlation Analysis Complete.")
        return group_dir

class CrossConditionAnalysis:
    
    def __init__(self, subject_results):
        self.results = subject_results
        
    def normalitycheck(self, z_random, z_sequential):
        """Select statistical test based on Shapiro-Wilk normality test."""
        try:
            _, p_rand = stats.shapiro(z_random)
            _, p_seq = stats.shapiro(z_sequential)
            # Use parametric if both are normal (p > 0.05)
            if p_rand > 0.05 and p_seq > 0.05:
                return 'ttest'
            else:
                return 'wilcoxon'
        except Exception:
            # Fallback to t-test if Shapiro-Wilk fails
            return 'ttest'
    
    def trianglecorn(self):
        pair_names = ['HFD-Lyapunov_Exponent', 'HFD-Phase_Volume', 
                       'Lyapunov_Exponent-Phase_Volume']
        
        seq_vectors = []
        rand_vectors = []
        
        for subj_res in self.results:
            rand_data = subj_res.get('finger_correlations', {}).get('Random', {})
            seq_data = subj_res.get('finger_correlations', {}).get('Sequential', {})
            
            seq_vec = []
            rand_vec = []
            
            for pair in pair_names:
                for f_class in sorted(rand_data.keys()):
                    z_seq = seq_data.get(f_class, {}).get('fisher_z', {}).get(pair)
                    z_rand = rand_data.get(f_class, {}).get('fisher_z', {}).get(pair)
                    
                    if z_seq is not None and z_rand is not None:
                        seq_vec.append(z_seq)
                        rand_vec.append(z_rand)
            
            if len(seq_vec) >= 3:
                seq_vectors.extend(seq_vec)
                rand_vectors.extend(rand_vec)
        
        if len(seq_vectors) < 3:
            return None, None
        
        r_stability, p_corr = stats.pearsonr(seq_vectors[:len(rand_vectors)], 
                                              rand_vectors[:len(seq_vectors)])
        return r_stability, p_corr
    
    def runana(self, output_dir):
        pair_names = ['HFD-Lyapunov_Exponent', 'HFD-Phase_Volume', 'Lyapunov_Exponent-Phase_Volume']
        
        print("\n" + "="*70)
        print("CROSS-CONDITION ANALYSIS: Sequential vs Random Effect Assessment")
        print("="*70)
        print("Goal: Identify FIXED BEHAVIOR (correlations stable across conditions)")
        print("-"*70)
        
        analysis_dir = os.path.join(output_dir, 'Cross_Condition_Analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        df_condition_effects = []
        df_fixed_behavior_summary = []
        df_subject_details = []
        
        n_total_subjects = len(self.results)
        
        for pair_name in pair_names:
            print(f"\n--- Metric Pair: {pair_name} ---")
            
            all_fingers = set()
            for subj_res in self.results:
                rand_data = subj_res.get('finger_correlations', {}).get('Random', {})
                seq_data = subj_res.get('finger_correlations', {}).get('Sequential', {})
                for f in rand_data.keys():
                    all_fingers.add(f)
                for f in seq_data.keys():
                    all_fingers.add(f)
            
            all_fingers = sorted(all_fingers)
            print(f"  Finger classes found: {all_fingers}")
            
            for f_class in all_fingers:
                z_random = []
                z_sequential = []
                subject_ids = []
                
                for subj_res in self.results:
                    subj_id = subj_res.get('subject', 'Unknown')
                    
                    rand_data = subj_res.get('finger_correlations', {}).get('Random', {})
                    seq_data = subj_res.get('finger_correlations', {}).get('Sequential', {})
                    
                    z_rand = rand_data.get(f_class, {}).get('fisher_z', {}).get(pair_name)
                    z_seq = seq_data.get(f_class, {}).get('fisher_z', {}).get(pair_name)
                    
                    if z_rand is not None and z_seq is not None:
                        z_random.append(z_rand)
                        z_sequential.append(z_seq)
                        subject_ids.append(subj_id)
                
                n_with_both = len(z_random)
                n_missing = n_total_subjects - n_with_both
                
                if n_with_both < 5:
                    print(f"  {f_class}: Insufficient paired data ({n_with_both}/{n_total_subjects})")
                    continue
                
                print(f"  {f_class}: {n_with_both} subjects with both conditions")
                
                diffs = np.array(z_sequential) - np.array(z_random)
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs, ddof=1)
                sem_diff = std_diff / np.sqrt(n_with_both)
                
                # --- SHAPIRO-WILK TEST SELECTION ---
                test_type = self.normalitycheck(z_random, z_sequential)
                
                if test_type == 'ttest':
                    t_stat, p_val = stats.ttest_rel(z_random, z_sequential)
                    test_name = "Paired t-test"
                else:
                    try:
                        t_stat, p_val = stats.wilcoxon(z_random, z_sequential)
                        test_name = "Wilcoxon signed-rank"
                    except Exception:
                        t_stat, p_val = stats.ttest_rel(z_random, z_sequential)
                        test_type = 'ttest'
                        test_name = "Paired t-test (fallback)"
                
  
                ci_low, ci_high = stats.t.interval(0.95, df=n_with_both-1, loc=mean_diff, scale=sem_diff)
                cohens_d = mean_diff / std_diff if std_diff > 0 else 0  # 
                
                is_fixed = p_val >= 0.05
                behavior_type = "FIXED" if is_fixed else "CONDITION-DEPENDENT"
                
                if not is_fixed:
                    direction = "Higher in Sequential" if mean_diff > 0 else "Higher in Random"
                else:
                    direction = "No significant difference"
                
                mean_r_random = np.tanh(np.mean(z_random))
                mean_r_sequential = np.tanh(np.mean(z_sequential))
                
                df_condition_effects.append({
                    'Metric_Pair': pair_name,
                    'Finger_Class': f_class,
                    'N_Subjects': n_with_both,
                    'N_Missing': n_missing,
                    'Test_Type': test_type,
                    'Mean_Z_Random': round(np.mean(z_random), 4),
                    'Mean_Z_Sequential': round(np.mean(z_sequential), 4),
                    'Mean_Diff_Z': round(mean_diff, 4),
                    'Std_Diff': round(std_diff, 4),
                    'CI_95_Low': round(ci_low, 4),
                    'CI_95_High': round(ci_high, 4),
                    'T_Statistic': round(t_stat, 4) if test_type == 'ttest' else None,
                    'W_Statistic': round(t_stat, 4) if test_type == 'wilcoxon' else None,
                    'P_Value': round(p_val, 6),
                    'Cohen_D': round(cohens_d, 4),
                    'Behavior_Type': behavior_type,
                    'Direction': direction,
                    'Mean_R_Random': round(mean_r_random, 4),
                    'Mean_R_Sequential': round(mean_r_sequential, 4),
                    'Fixed_Behavior': 'Yes' if is_fixed else 'No'
                })
                
                df_fixed_behavior_summary.append({
                    'Metric_Pair': pair_name,
                    'Finger_Class': f_class,
                    'Behavior_Type': behavior_type,
                    'Fixed_Behavior': 'Yes' if is_fixed else 'No',
                    'Test_Type': test_type,
                    'Effect_Size_Cohen_D': round(cohens_d, 4),
                    'P_Value': round(p_val, 6)
                })
                
                for i, subj_id in enumerate(subject_ids):
                    df_subject_details.append({
                        'Metric_Pair': pair_name,
                        'Finger_Class': f_class,
                        'Subject_ID': subj_id,
                        'Z_Random': round(z_random[i], 4),
                        'Z_Sequential': round(z_sequential[i], 4),
                        'Difference_Z': round(diffs[i], 4)
                    })
                
                sig_mark = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                print(f"    Test: {test_name}, Diff Z: {mean_diff:.3f} [{ci_low:.3f}, {ci_high:.3f}], p={p_val:.4f} {sig_mark}")
                print(f"    Behavior: {behavior_type} (Cohen's d = {cohens_d:.3f})")
        
        # --- SAVE EFFECTS FILES ---
        if df_condition_effects:
            df_out = pd.DataFrame(df_condition_effects)
            path = os.path.join(analysis_dir, 'Cross_Condition_Effects_Main.csv')
            df_out.to_csv(path, index=False)
            print(f"\n[OK] Saved main results: {path}")
        
        if df_fixed_behavior_summary:
            df_out = pd.DataFrame(df_fixed_behavior_summary)
            path = os.path.join(analysis_dir, 'Fixed_Behavior_Summary.csv')
            df_out.to_csv(path, index=False)
            print(f"[OK] Saved fixed behavior summary: {path}")
        
        if df_subject_details:
            df_out = pd.DataFrame(df_subject_details)
            path = os.path.join(analysis_dir, 'Subject_Level_Details.csv')
            df_out.to_csv(path, index=False)
            print(f"[OK] Saved subject-level details: {path}")
        
        # --- UPPER-TRIANGLE MATRIX CORRELATION ---
        r_stab, p_corr = self.trianglecorn()
        
        with open(os.path.join(analysis_dir, 'Upper_Triangle_Matrix_Correlation.txt'), 'w') as f:
            f.write("Upper-Triangle Matrix Correlation (Chaos Domain)\n")
            f.write("=============================================\n\n")
            f.write(f"r_stability = {r_stab:.4f}\n") if r_stab else f.write("r_stability = N/A\n")
            f.write(f"P-value = {p_corr:.6f}\n") if p_corr else f.write("P-value = N/A\n")
            f.write(f"\nMethod: Pearson correlation between Fisher-Z vectors\n")
            f.write(f"extracted from upper-triangles of correlation matrices\n")
            f.write(f"(3 metric pairs x 5 fingers x N subjects per condition).\n")
        
        print(f"\n  ✓ Chaos Stability r = {r_stab:.4f} (p = {p_corr:.6f})")
        
        # --- PRINT OVERALL SUMMARY ---
        print("\n" + "-"*70)
        print("OVERALL SUMMARY: Fixed vs Condition-Dependent Behaviors")
        print("-"*70)
        
        df_summary = pd.DataFrame(df_fixed_behavior_summary) if df_fixed_behavior_summary else None
        if df_summary is not None and len(df_summary) > 0:
            fixed_count = len(df_summary[df_summary['Fixed_Behavior'] == 'Yes'])
            total_count = len(df_summary)
            print(f"\nTotal finger/metric combinations tested: {total_count}")
            print(f"Fixed behaviors (stable across conditions): {fixed_count} ({fixed_count/total_count*100:.1f}%)")
            print(f"Condition-dependent behaviors: {total_count - fixed_count} ({(total_count-fixed_count)/total_count*100:.1f}%)")
            
            for pair in df_summary['Metric_Pair'].unique():
                pair_data = df_summary[df_summary['Metric_Pair'] == pair]
                pair_fixed = len(pair_data[pair_data['Fixed_Behavior'] == 'Yes'])
                pair_total = len(pair_data)
                print(f"  {pair}: {pair_fixed}/{pair_total} fixed ({pair_fixed/pair_total*100:.1f}%)")
            
            test_counts = df_summary['Test_Type'].value_counts()
            print(f"\n  Test selection: {test_counts.to_dict()}")
        else:
            print("[WARN] No results to summarise.")
        
        print(f"\n  Chaos Stability r = {r_stab:.4f} (p = {p_corr:.6f})" if r_stab else "\n  Chaos Stability r = N/A")
        print("\n[DONE] Cross-Condition Analysis Complete.")
        chr_stab = r_stab

        return analysis_dir, chr_stab
    

class ChannelLevelEffectSizeAnalysis:
    """Compute per-channel Cohen's d for Linear vs Chaos features"""
    
    def __init__(self, df_merged):
        self.df = df_merged
    
    def runana(self, output_dir):
        print("\n[CHANNEL LEVEL EFFECT SIZES] Sequential vs Random")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # WITH THIS:
        all_channels = sorted(self.df['Channel'].unique())  # Use all available!
        target_channels = all_channels
        
        features_to_analyse = {
            'Chaos': ['HFD', 'Lyapunov_Exponent', 'Phase_Volume'],
            'Linear': [c for c in self.df.columns if c.startswith('Lin_')]
        }
        
        results = []
        
        # NEW: Track counts for threshold analysis
        threshold = 0.15
        domain_counts = {}
        
        for domain, feats in features_to_analyse.items():
            print(f"\n  Processing {domain} domain: {len(feats)} features")
            
            # Initialize counters for this domain
            domain_counts[domain] = {
                'total': 0,
                'negligible': 0,      # |d| < 0.15
                'small': 0,            # 0.15 ≤ |d| < 0.3
                'medium': 0,           # 0.3 ≤ |d| < 0.5
                'large': 0             # |d| ≥ 0.5
            }
            
            for channel in target_channels:
                if channel not in self.df['Channel'].unique():
                    continue
                
                needed_cols = feats + ['Condition', 'Channel']
                ch_df = self.df[self.df['Channel'] == channel][needed_cols].dropna()
                
                if len(ch_df) < 20:
                    continue
                
                seq_data = ch_df[ch_df['Condition'] == 'Sequential'][feats]
                rand_data = ch_df[ch_df['Condition'] == 'Random'][feats]
                
                for feat in feats:
                    seq_vals = seq_data[feat].values
                    rand_vals = rand_data[feat].values
                    
                    if len(seq_vals) < 5 or len(rand_vals) < 5:
                        continue
                    
                    mean1, std1 = np.mean(seq_vals), np.std(seq_vals, ddof=1)
                    mean2, std2 = np.mean(rand_vals), np.std(rand_vals, ddof=1)
                    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                    
                    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                    abs_cohens_d = abs(cohens_d)
                    
                    # Update counters based on threshold
                    domain_counts[domain]['total'] += 1
                    
                    if abs_cohens_d < 0.15:
                        domain_counts[domain]['negligible'] += 1
                    elif abs_cohens_d < 0.3:
                        domain_counts[domain]['small'] += 1
                    elif abs_cohens_d < 0.5:
                        domain_counts[domain]['medium'] += 1
                    else:
                        domain_counts[domain]['large'] += 1
                    
                    results.append({
                        'Domain': domain,
                        'Channel': channel,
                        'Feature': feat,
                        'N_Sequential': len(seq_vals),
                        'N_Random': len(rand_vals),
                        'Mean_Sequential': round(mean1, 4),
                        'Mean_Random': round(mean2, 4),
                        'Pooled_SD': round(pooled_std, 4),
                        'Cohens_D': round(cohens_d, 4),
                        'Abs_Cohens_D': round(abs_cohens_d, 4),
                        'Effect_Category': 'Negligible' if abs_cohens_d < 0.15 else 
                                          'Small' if abs_cohens_d < 0.3 else
                                          'Medium' if abs_cohens_d < 0.5 else 
                                          'Large'
                    })
        
        if results:
            df_out = pd.DataFrame(results)
            path = os.path.join(output_dir, 'Per_Channel_Effect_Sizes.csv')
            df_out.to_csv(path, index=False)
            print(f"[OK] Saved to {path}")
            
            # NEW: Print threshold summary table
            print("\n" + "="*70)
            print("EFFECT SIZE THRESHOLD SUMMARY (|d| thresholds)")
            print("="*70)
            print(f"\nThreshold for Negligible Effect: |d| < {threshold}")
            print("-"*70)
            print(f"{'Domain':<12} {'Total':>8} {'Negligible':>12} {'Small':>8} {'Medium':>8} {'Large':>8} {'% Below Threshold':>16}")
            print("-"*70)
            
            for domain in ['Chaos', 'Linear']:
                dom_stats = domain_counts[domain]
                total = dom_stats['total']
                neg = dom_stats['negligible']
                smal = dom_stats['small']
                med = dom_stats['medium']
                lg = dom_stats['large']
                
                pct_below = (neg / total * 100) if total > 0 else 0
                
                print(f"{domain:<12} {total:>8} {neg:>12} ({pct_below:>5.1f}%) {smal:>8} {med:>8} {lg:>8} {pct_below:>15.1f}%")
            
            print("-"*70)
            
            # NEW: Highlight key insight for Linear specifically
            lin_neg = domain_counts['Linear']['negligible']
            lin_total = domain_counts['Linear']['total']
            lin_pct = (lin_neg / lin_total * 100) if lin_total > 0 else 0
            
            print(f"\n→ KEY INSIGHT:")
            print(f"  Linear domain: {lin_neg}/{lin_total} features ({lin_pct:.1f}%) have negligible condition effects (|d| < 0.15)")
            
            if lin_pct > 50:
                print(f"  → Majority of Linear features show minimal Sequential vs Random differences")
            elif lin_pct > 30:
                print(f"  → Substantial portion of Linear features show minimal Sequential vs Random differences")
            else:
                print(f"  → Most Linear features exhibit meaningful condition-dependent variation")
            
            cha_neg = domain_counts['Chaos']['negligible']
            cha_total = domain_counts['Chaos']['total']
            cha_pct = (cha_neg / cha_total * 100) if cha_total > 0 else 0
            
            print(f"\n  Chaos domain: {cha_neg}/{cha_total} features ({cha_pct:.1f}%) have negligible condition effects (|d| < 0.15)")
            
            diff = lin_pct - cha_pct
            if diff > 0:
                print(f"  → Linear has {diff:.1f}% MORE negligible effects than Chaos")
            elif diff < 0:
                print(f"  → Chaos has {-diff:.1f}% MORE negligible effects than Linear")
            
            print("="*70)
        
        return results, domain_counts
    
def stability(cha_results, df_merged, output_dir):

    print("\n[INFO] Using pre-computed finger_correlations...")
    
    pair_names = ['HFD-Lyapunov_Exponent', 'HFD-Phase_Volume', 'Lyapunov_Exponent-Phase_Volume']
    seq_vectors_cha = []
    rand_vectors_cha = []
    
    n_chaos_valid = 0
    for subj_res in cha_results:
        rand_data = subj_res.get('finger_correlations', {}).get('Random', {})
        seq_data = subj_res.get('finger_correlations', {}).get('Sequential', {})
        
        subj_has_data = False
        for pair in pair_names:
            for f_class in sorted(rand_data.keys()):
                z_seq = seq_data.get(f_class, {}).get('fisher_z', {}).get(pair)
                z_rand = rand_data.get(f_class, {}).get('fisher_z', {}).get(pair)
                
                if z_seq is not None and z_rand is not None:
                    seq_vectors_cha.append(z_seq)
                    rand_vectors_cha.append(z_rand)
                    subj_has_data = True
        
        if subj_has_data:
            n_chaos_valid += 1
    
    min_len_cha = min(len(seq_vectors_cha), len(rand_vectors_cha))
    if min_len_cha >= 3:
        r_chaos, p_chaos = stats.pearsonr(seq_vectors_cha[:min_len_cha], rand_vectors_cha[:min_len_cha])
        cha_result = {
            'Domain': 'Chaos',
            'R_Stability': round(r_chaos, 4),
            'P_Value': f"{p_chaos:.2e}",
            'Vector_Elements': min_len_cha,
            'Subjects_Valid': n_chaos_valid,
            'Meets_Threshold_0.8': 'Yes' if r_chaos >= 0.8 else 'No'
        }
        print(f"\n  Chaos Stability:")
        print(f"    Subjects: {n_chaos_valid}")
        print(f"    Vector Elements: {min_len_cha}")
        print(f"    r = {r_chaos:.4f} (p = {p_chaos:.2e})")
        print(f"    Threshold (r > 0.8): {'YES' if r_chaos >= 0.8 else 'NO'}")
    else:
        cha_result = None
        print("  [ERROR] Insufficient chaos data")
    
    # ==========================================
    # LINEAR STABILITY (SAME methodology adapted)
    # ==========================================
    print("\n[LINEAR STABILITY] Computing using identical pipeline...")
    
    # We need to compute similar Finger Correlations for Linear features
    # Since df_merged has per-channel spectral bands, we treat each band like a "metric"
    
    lin_features = [c for c in df_merged.columns if c.startswith('Lin_')]
    if len(lin_features) < 2:
        print("  [SKIP] Insufficient linear features")
        lin_result = None
    else:
        seq_vectors_lin = []
        rand_vectors_lin = []
        n_lin_valid = 0
        
        subjects = sorted(df_merged['Subject'].unique())
        fingers = sorted(set(str(c) for c in df_merged['Class'].unique()))
        
        for subj in subjects:
            subj_df = df_merged[df_merged['Subject'] == subj]
            
            subj_has_data = False
            
            for finger in fingers:
                f_df = subj_df[subj_df['Class'].astype(str) == str(finger)]
                
                # Compute feature-feature correlations LIKE chaos does
                for cond_label, vec_store in [('Sequential', seq_vectors_lin), ('Random', rand_vectors_lin)]:
                    cond_data = f_df[f_df['Condition'] == cond_label][lin_features].dropna(axis=1)
                    
                    if len(cond_data.columns) < 2 or len(cond_data) < 10:
                        continue
                    
                    try:
                        corr_matrix = cond_data.corr()
                        n_feats = len(corr_matrix)
                        vec = corr_matrix.values[np.triu_indices(n_feats, k=1)]
                        valid = ~np.isnan(vec)
                        vec_clean = vec[valid]
                        
                        if len(vec_clean) >= 1:
                            z_vals = [Utils.fisher_z(r) for r in vec_clean]
                            vec_store.extend(z_vals)
                            subj_has_data = True
                    except Exception:
                        continue
            
            if subj_has_data:
                n_lin_valid += 1
        
        min_len_lin = min(len(seq_vectors_lin), len(rand_vectors_lin))
        if min_len_lin >= 3:
            r_linear, p_linear = stats.pearsonr(seq_vectors_lin[:min_len_lin], rand_vectors_lin[:min_len_lin])
            lin_result = {
                'Domain': 'Linear',
                'R_Stability': round(r_linear, 4),
                'P_Value': f"{p_linear:.2e}",
                'Vector_Elements': min_len_lin,
                'Subjects_Valid': n_lin_valid,
                'Meets_Threshold_0.8': 'Yes' if r_linear >= 0.8 else 'No'
            }
            print(f"\n  Linear Stability:")
            print(f"    Subjects: {n_lin_valid}")
            print(f"    Vector Elements: {min_len_lin}")
            print(f"    r = {r_linear:.4f} (p = {p_linear:.2e})")
            print(f"    Threshold (r > 0.8): {'Yes' if r_linear >= 0.8 else 'No'}")
        else:
            lin_result = None
            print("  [ERROR] Insufficient linear data")
    
    # ==========================================
    # SAVE COMPARISON & PRINT SUMMARY
    # ==========================================
    stab_dir = os.path.join(output_dir, 'Unified_Stability_Script2_Method')
    os.makedirs(stab_dir, exist_ok=True)
    
    if lin_result and cha_result:
        df_compare = pd.DataFrame([cha_result, lin_result])
        csv_path = os.path.join(stab_dir, 'Stability_Comparison.csv')
        df_compare.to_csv(csv_path, index=False)
        
        print("\n" + "-"*70)
        print("FINAL COMPARISON")
        print("-"*70)
        print(f"  Domain       |   R-Stability   |  Threshold Met  ")
        print(f"  {'-'*60}")
        print(f"  {'Chaos':<14}| {cha_result['R_Stability']:>12} |     {cha_result['Meets_Threshold_0.8']:<15}")
        print(f"  {'Linear':<14}| {lin_result['R_Stability']:>12} |     {lin_result['Meets_Threshold_0.8']:<15}")
        print(f"  {'-'*60}")
        
        diff = cha_result['R_Stability'] - lin_result['R_Stability']
        print(f"\n  Difference (Chaos - Linear): {diff:+.4f}")
        
        if abs(diff) > 0.1:
            print(f"  → LARGE GAP: Domains show genuinely different stability patterns")
        elif diff > 0:
            print(f"  → Slightly higher chaos stability")
        else:
            print(f"  → Slightly higher linear stability")
    else:
        print("  [WARN] Could not complete comparison due to missing data")
    
    return cha_result, lin_result

class LinearCalcova:
    """Mirror of subjectwise.calcova() but for Linear/Spectral features"""
    
    @staticmethod
    def calcova(df_linear, subject_name):
        """
        Perform identical correlation analysis on linear spectral features
        Returns the SAME dictionary structure as subjectwise.calcova()
        """
        if df_linear is None or len(df_linear) == 0:
            return None
        
        # Get all spectral band columns
        # In LinearCalcova.calcova():
        target_bands = ['Lin_theta', 'Lin_alpha', 'Lin_beta']
        lin_features = [c for c in df_linear.columns if c.startswith('Lin_') and c in target_bands]
        if len(lin_features) < 2:
            print(f"[WARN] Insufficient linear features for {subject_name}")
            return None
        
        # Store feature names as "pseudo-metrics" for downstream compatibility
        metrics_as_strings = [f.replace('Lin_', '') for f in lin_features]
        
        unique_classes = sorted([str(c) for c in df_linear['Class'].unique()])
        
        results = {
            'subject': subject_name,
            'finger_correlations': {'Random': {}, 'Sequential': {}},
            'finger_shifts': {'Random': {}, 'Sequential': {}},
            'channel_finger_diff': {'Random': {}, 'Sequential': {}},
            'multivariate': {'Random': {}, 'Sequential': {}},
            'raw_data_cache': {},  # Store full feature vectors per finger
            'feature_names': metrics_as_strings  # Track which bands were used
        }
        
        def process(cond_df, cond_label):
            if len(cond_df) == 0:
                return
            
            # --- CACHE RAW DATA FOR SUBSEQUENT MULTIVARIATE TESTS ---
            for cls in unique_classes:
                cls_df = cond_df[cond_df['Class'] == str(cls)]
                clean = cls_df[lin_features].dropna(axis=1)  # Drop bands with any NaN
                
                if len(clean) >= 10:
                    cache_key = f"{cond_label}_{cls}"
                    results['raw_data_cache'][cache_key] = clean.reset_index(drop=True)
            
            # --- FINGER-LEVEL CORRELATION ANALYSIS ---
            finger_r_data = {cls: {} for cls in unique_classes}
            
            # Build all pairwise combinations of spectral bands
            n_bands = len(lin_features)
            pairs_list = [(lin_features[i], lin_features[j]) 
                          for i in range(n_bands) 
                          for j in range(i + 1, n_bands)]
            pair_names = [f"{p[0].replace('Lin_', '')}-{p[1].replace('Lin_', '')}" 
                          for p in pairs_list]
            
            for cls in unique_classes:
                cls_df = cond_df[cond_df['Class'] == str(cls)]
                clean = cls_df[lin_features].dropna(axis=1)
                
                if len(clean) < 20:
                    continue
                
                n_trials = len(clean)
                finger_r_data[cls]['n'] = n_trials
                finger_r_data[cls]['raw_corr'] = {}
                finger_r_data[cls]['fisher_z'] = {}
                finger_r_data[cls]['se'] = {}
                
                for feat_i, feat_j in pairs_list:
                    x = clean[feat_i].values
                    y = clean[feat_j].values
                    
                    if np.std(x) > 0 and np.std(y) > 0:
                        r_val = np.corrcoef(x, y)[0, 1]
                        
                        if not np.isnan(r_val):
                            z_val = Utils.fisher_z(r_val)
                            se_val = 1.0 / np.sqrt(n_trials - 3)
                            
                            # Use shortened band names for consistency with chaos output
                            short_i = feat_i.replace('Lin_', '')
                            short_j = feat_j.replace('Lin_', '')
                            pair_key = f"{short_i}-{short_j}"
                            
                            finger_r_data[cls]['raw_corr'][pair_key] = float(r_val)
                            finger_r_data[cls]['fisher_z'][pair_key] = float(z_val)
                            finger_r_data[cls]['se'][pair_key] = float(se_val)
                
                # --- PARTIAL CORRELATIONS (Same method as chaos) ---
                try:
                    corr_matrix = clean.corr().values
                    diag = np.diag_indices(len(lin_features))
                    
                    # For each band, compute partials with all others controlling for remaining
                    partial_dict = {}
                    n = len(lin_features)
                    
                    if n >= 3:
                        # Compute inverse correlation matrix for partials
                        corr_inv = np.linalg.inv(corr_matrix + np.eye(n) * 1e-6)
                        
                        # Extract partial correlations from inverse matrix
                        for i in range(n):
                            for j in range(i + 1, n):
                                short_i = lin_features[i].replace('Lin_', '')
                                short_j = lin_features[j].replace('Lin_', '')
                                
                                r_partial = -corr_inv[i, j] / np.sqrt(corr_inv[i, i] * corr_inv[j, j])
                                r_partial = np.clip(r_partial, -0.999, 0.999)
                                
                                partial_dict[f"{short_i}-{short_j}"] = float(r_partial)
                    
                    finger_r_data[cls]['partial_corr'] = partial_dict
                    
                except np.linalg.LinAlgError:
                    finger_r_data[cls]['partial_corr'] = {}
                
                # --- MULTIPLE R² VALUES (Same method as chaos) ---
                r_sq_vals = {}
                try:
                    # For each band, predict from all other bands
                    for i, target_feat in enumerate(lin_features):
                        predictors = [f for j, f in enumerate(lin_features) if j != i]
                        
                        y = clean[target_feat].values
                        X = sm.add_constant(clean[predictors].values)
                        model = sm.OLS(y, X).fit()
                        
                        short_target = target_feat.replace('Lin_', '')
                        pred_str = '+'.join([p.replace('Lin_', '') for p in predictors])
                        r_sq_vals[f"{short_target}_Predicted_by_{pred_str}"] = float(model.rsquared)
                        
                except Exception:
                    pass
                
                finger_r_data[cls]['multi_r2'] = r_sq_vals
            
            results['finger_correlations'][cond_label] = finger_r_data
            
            # --- FINGER SHIFTS (Identical to chaos) ---
            finger_classes = sorted([c for c in finger_r_data.keys() 
                                    if 'fisher_z' in finger_r_data[c]])
            
            for i in range(len(finger_classes)):
                for j in range(i + 1, len(finger_classes)):
                    f_i = finger_classes[i]
                    f_j = finger_classes[j]
                    comp_key = f"{f_i}_vs_{f_j}"
                    
                    if comp_key not in results['finger_shifts'][cond_label]:
                        results['finger_shifts'][cond_label][comp_key] = {}
                    
                    for pair_key in pair_names:
                        z_i = finger_r_data[f_i].get('fisher_z', {}).get(pair_key)
                        z_j = finger_r_data[f_j].get('fisher_z', {}).get(pair_key)
                        se_i = finger_r_data[f_i].get('se', {}).get(pair_key)
                        se_j = finger_r_data[f_j].get('se', {}).get(pair_key)
                        
                        if z_i is None or z_j is None or se_i is None or se_j is None:
                            continue
                        
                        diff_z = z_i - z_j
                        se_diff = np.sqrt(se_i**2 + se_j**2)
                        
                        if se_diff > 0:
                            z_stat = diff_z / se_diff
                            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        else:
                            z_stat = 0
                            p_val = 1.0
                        
                        r_i = finger_r_data[f_i].get('raw_corr', {}).get(pair_key, 0)
                        r_j = finger_r_data[f_j].get('raw_corr', {}).get(pair_key, 0)
                        
                        results['finger_shifts'][cond_label][comp_key][pair_key] = {
                            'diff_z': float(diff_z),
                            'p_value': float(p_val),
                            'r_i': float(r_i),
                            'r_j': float(r_j)
                        }
            
            # --- CHANNEL-LEVEL FINGER DIFFERENCES ---
            if 'Channel' in cond_df.columns:
                channels_present = cond_df['Channel'].unique()
                
                for ch in channels_present:
                    ch_df = cond_df[cond_df['Channel'] == ch]
                    
                    if ch not in results['channel_finger_diff'][cond_label]:
                        results['channel_finger_diff'][cond_label][ch] = {}
                    
                    # Per-finger correlations within this channel
                    ch_finger_corr = {}
                    for cls in unique_classes:
                        cls_ch_df = ch_df[ch_df['Class'] == str(cls)]
                        clean_ch = cls_ch_df[lin_features].dropna(axis=1)
                        
                        if len(clean_ch) < 20:
                            continue
                        
                        ch_finger_corr[cls] = {}
                        for feat_i, feat_j in pairs_list:
                            if np.std(clean_ch[feat_i]) > 0 and np.std(clean_ch[feat_j]) > 0:
                                r_val = clean_ch[feat_i].corr(clean_ch[feat_j])
                                
                                if not np.isnan(r_val):
                                    short_i = feat_i.replace('Lin_', '')
                                    short_j = feat_j.replace('Lin_', '')
                                    ch_finger_corr[cls][f"{short_i}-{short_j}"] = r_val
                    
                    # Compare finger pairs within this channel
                    finger_keys = sorted(ch_finger_corr.keys())
                    for i in range(len(finger_keys)):
                        for j in range(i + 1, len(finger_keys)):
                            f_i = finger_keys[i]
                            f_j = finger_keys[j]
                            comp_key = f"{f_i}_vs_{f_j}"
                            
                            for pair_key in pair_names:
                                if pair_key not in ch_finger_corr[f_i] or pair_key not in ch_finger_corr[f_j]:
                                    continue
                                
                                r_i = ch_finger_corr[f_i][pair_key]
                                r_j = ch_finger_corr[f_j][pair_key]
                                
                                z_i = Utils.fisher_z(r_i)
                                z_j = Utils.fisher_z(r_j)
                                
                                n_i = len(ch_df[ch_df['Class'] == str(f_i)][lin_features].dropna())
                                n_j = len(ch_df[ch_df['Class'] == str(f_j)][lin_features].dropna())
                                
                                se_diff = np.sqrt(1/(n_i-3) + 1/(n_j-3))
                                diff_z = z_i - z_j
                                
                                if se_diff > 0:
                                    z_stat = diff_z / se_diff
                                    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                                else:
                                    z_stat = 0
                                    p_val = 1.0
                                
                                direction = f"{f_i}>{f_j}" if diff_z > 0 else f"{f_j}>{f_i}"
                                significant = 'Yes' if p_val < 0.05 else 'No'
                                
                                if pair_key not in results['channel_finger_diff'][cond_label][ch]:
                                    results['channel_finger_diff'][cond_label][ch][pair_key] = {}
                                
                                results['channel_finger_diff'][cond_label][ch][pair_key][comp_key] = {
                                    'winner': f_i if diff_z > 0 else f_j,
                                    'direction': direction,
                                    'mean_diff': float(diff_z),
                                    'abs_diff': float(abs(diff_z)),
                                    'p_value': float(p_val),
                                    'z_stat': float(z_stat),
                                    'significant': significant,
                                    'r_i': float(r_i),
                                    'r_j': float(r_j)
                                }
            
            # Populate multivariate dict
            results['multivariate'][cond_label] = {}
            for cls in unique_classes:
                if cls in finger_r_data:
                    results['multivariate'][cond_label][cls] = {
                        'partial': finger_r_data[cls].get('partial_corr', {}),
                        'multi_r2': finger_r_data[cls].get('multi_r2', {}),
                        'n': finger_r_data[cls].get('n', 0)
                    }
        
        # Run for both conditions
        for cond in ['Random', 'Sequential']:
            cond_df = df_linear[df_linear['Condition'] == cond]
            process(cond_df, cond)
        
        return results

def compare_domain_stabilities(cha_results, lin_results, output_dir):
    """
    Compare stability between chaos and linear domains with statistical testing
    """
    pair_names = ['HFD-Lyapunov_Exponent', 'HFD-Phase_Volume', 'Lyapunov_Exponent-Phase_Volume']
    
    # Extract linear pair names dynamically
    if lin_results:
        first_lin = lin_results[0]
        first_cond = first_lin.get('finger_correlations', {}).get('Sequential', {})
        first_finger = list(first_cond.keys())[0] if first_cond else None
        if first_finger:
            lin_pairs = list(first_cond[first_finger].get('fisher_z', {}).keys())
        else:
            lin_pairs = []
    else:
        lin_pairs = []
    
    # === EXTRACT ALL FISHER-Z VALUES FOR BOTH DOMAINS ===
    cha_seq_vectors = []
    cha_rand_vectors = []
    lin_seq_vectors = []
    lin_rand_vectors = []
    
    n_subj_cha = len(cha_results)
    n_subj_lin = len(lin_results)
    
    for subj_idx, (cha_res, lin_res) in enumerate(zip(cha_results, lin_results)):
        # CHAOS
        cha_rand = cha_res.get('finger_correlations', {}).get('Random', {})
        cha_seq = cha_res.get('finger_correlations', {}).get('Sequential', {})
        
        for cls in cha_rand.keys():
            for pair in pair_names:
                z_rand = cha_rand[cls].get('fisher_z', {}).get(pair)
                z_seq = cha_seq[cls].get('fisher_z', {}).get(pair)
                
                if z_rand is not None and z_seq is not None:
                    cha_rand_vectors.append(z_rand)
                    cha_seq_vectors.append(z_seq)
        
        # LINEAR
        lin_rand = lin_res.get('finger_correlations', {}).get('Random', {})
        lin_seq = lin_res.get('finger_correlations', {}).get('Sequential', {})
        
        for cls in lin_rand.keys():
            for pair in lin_pairs:
                z_rand = lin_rand[cls].get('fisher_z', {}).get(pair)
                z_seq = lin_seq[cls].get('fisher_z', {}).get(pair)
                
                if z_rand is not None and z_seq is not None:
                    lin_rand_vectors.append(z_rand)
                    lin_seq_vectors.append(z_seq)
    
    print(f"\nExtracted {len(cha_seq_vectors)} chaos Z-values, {len(lin_seq_vectors)} linear Z-values")
    print(f"Across {n_subj_cha} chaos subjects, {n_subj_lin} linear subjects")
    
    if len(cha_seq_vectors) < 5 or len(lin_seq_vectors) < 5:
        print("[ERROR] Insufficient data for comparison")
        return None
    
    # === COMPUTE STABILITY R-FOR EACH DOMAIN ===
    r_chaos, p_chaos = stats.pearsonr(cha_seq_vectors, cha_rand_vectors)
    r_linear, p_linear = stats.pearsonr(lin_seq_vectors, lin_rand_vectors)
    
    print(f"\nDomain Stability Results:")
    print(f"  Chaos:   r = {r_chaos:.4f}, p = {p_chaos:.2e}")
    print(f"  Linear:  r = {r_linear:.4f}, p = {p_linear:.2e}")
    print(f"  Difference: Δr = {r_chaos - r_linear:+.4f}")
    
    # === TEST IF DIFFERENCE IS SIGNIFICANT USING FISHER-R-TO-Z ===
    # Convert both r-values to Fisher-Z
    z_chaos = Utils.fisher_z(r_chaos)
    z_linear = Utils.fisher_z(r_linear)
    
    # SE depends on vector lengths
    se_diff = np.sqrt(1/(len(cha_seq_vectors) - 3) + 1/(len(lin_seq_vectors) - 3))
    z_stat_diff = abs(z_chaos - z_linear) / se_diff
    p_diff = 2 * (1 - stats.norm.cdf(z_stat_diff))
    
    ci_low = (z_chaos - z_linear) - 1.96 * se_diff
    ci_high = (z_chaos - z_linear) + 1.96 * se_diff
    
    print(f"\nSignificance Test:")
    print(f"  Z-statistic (difference): {z_stat_diff:.3f}")
    print(f"  p-value: {p_diff:.4f}")
    print(f"  95% CI for Δr (Fisher-Z scale): [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Is Chaos more stable? {'YES' if p_diff < 0.05 and r_chaos > r_linear else 'NO'}")
    
    # === SAVE RESULTS ===
    stab_dir = os.path.join(output_dir, 'Domain_Comparison_FisherZ_Test')
    os.makedirs(stab_dir, exist_ok=True)
    
    results_df = pd.DataFrame([{
        'Domain': 'Chaos',
        'R_Stability': round(r_chaos, 4),
        'P_Value': f"{p_chaos:.2e}",
        'N_Vector_Elements': len(cha_seq_vectors),
        'N_Subjects': n_subj_cha,
        'Meets_Threshold_0.8': 'Yes' if r_chaos >= 0.8 else 'No'
    }, {
        'Domain': 'Linear',
        'R_Stability': round(r_linear, 4),
        'P_Value': f"{p_linear:.2e}",
        'N_Vector_Elements': len(lin_seq_vectors),
        'N_Subjects': n_subj_lin,
        'Meets_Threshold_0.8': 'Yes' if r_linear >= 0.8 else 'No'
    }])
    
    comparison_df = pd.DataFrame([{
        'Comparison': 'Chaos vs Linear',
        'Δ_R': round(r_chaos - r_linear, 4),
        'Fisher_Z_Difference': round(z_chaos - z_linear, 4),
        'SE_Difference': round(se_diff, 4),
        'Z_Statistic': round(z_stat_diff, 3),
        'P_Value_Diff': round(p_diff, 6),
        'CI_95_Low': round(ci_low, 4),
        'CI_95_High': round(ci_high, 4),
        'Significant_at_p0.05': 'Yes' if p_diff < 0.05 else 'No',
        'Chaos_More_Stable': 'Yes' if r_chaos > r_linear and p_diff < 0.05 else 'No'
    }])
    
    results_df.to_csv(os.path.join(stab_dir, 'Domain_Stability_Results.csv'), index=False)
    comparison_df.to_csv(os.path.join(stab_dir, 'Domain_Comparison_Test.csv'), index=False)
    
    return {
        'r_chaos': r_chaos,
        'r_linear': r_linear,
        'p_diff': p_diff,
        'is_significant': p_diff < 0.05,
        'results_df': results_df,
        'comparison_df': comparison_df
    }


def plot_centroid_attractors(df_chaos, output_dir, n_subjects=20):
    """
    Uses Phase_Centroid_X/Y/Z from fanalysis() as the 3D axes
    Shows spatial distribution of attractor centers across trials
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify the columns exist
    required_cols = ['Phase_Centroid_X', 'Phase_Centroid_Y', 'Phase_Centroid_Z']
    missing = [c for c in required_cols if c not in df_chaos.columns]
    
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        print("These must come from fanalysis() function")
        return
    
    unique_classes = sorted(df_chaos['Class'].unique())[:5]
    unique_conditions = df_chaos['Condition'].unique()
    
    finger_colors = {3: '#FF6B6B', 4: '#FFA94D', 5: '#20C997', 6: '#4DABF7', 7: '#333366'}
    
    for condition in unique_conditions:
        cond_df = df_chaos[df_chaos['Condition'] == condition]
        
        for cls in unique_classes:
            finger_df = cond_df[cond_df['Class'] == cls]
            
            if len(finger_df) < 100:
                continue
            
            fig = plt.figure(figsize=(13, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            color = finger_colors.get(int(cls), 'steelblue')
            ax.scatter(
                finger_df['Phase_Centroid_X'].values,
                finger_df['Phase_Centroid_Y'].values,
                finger_df['Phase_Centroid_Z'].values,
                c=color, s=35, alpha=0.6, edgecolors='none'
            )
            
            # Add HFD and Lyapunov as color/intensity dimension
            hm = finger_df['HFD'].mean()
            lm = finger_df['Lyapunov_Exponent'].mean()
            pv = finger_df['Phase_Volume'].mean()
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            stat_text = f"Cluster: {cls}_{condition}\nN={len(finger_df)}\nHFD={hm:.2f}\nLyap={lm:.4f}\nVol={pv:.2f}"
            ax.text2D(0.05, 0.95, stat_text, transform=ax.transAxes,
                     fontsize=11, verticalalignment='top', bbox=props)
            
            ax.set_xlabel('Phase Centroid X', fontsize=12)
            ax.set_ylabel('Phase Centroid Y', fontsize=12)
            ax.set_zlabel('Phase Centroid Z', fontsize=12)
            ax.set_title(f'Finger {cls} | {condition}', fontsize=18, fontweight='bold')
            ax.view_init(elev=25, azim=-60)
            ax.grid(True, alpha=0.3)
            
            filepath = os.path.join(output_dir, f'PhaseCentroid_Attractor_Finger{cls}_{condition}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✓ Finger {cls} | {condition}: {len(finger_df)} points")
    
    return output_dir


def apply_in_memory_log_transform(df_chaos):
    """
    SAFE wrapper: Apply log transformation ONLY if Phase Volume exceeds threshold.
    Returns modified copy without affecting original dataframe.
    """
    
    pv_max = df_chaos['Phase_Volume'].max()
    
    if pv_max > 1e6:
        print(f"\n{'='*70}")
        print(f"⚠️  IN-MEMORY PHASE VOLUME FIX APPLIED")
        print(f"   Original max: {pv_max:.2e}")
        df_corrected = df_chaos.copy()
        df_corrected['Phase_Volume'] = np.where(
            df_corrected['Phase_Volume'] > 0,
            np.log10(df_corrected['Phase_Volume']),
            0.0
        )
        print(f"   Corrected max: {df_corrected['Phase_Volume'].max():.2f}")
        print(f"   This transformation is IN-MEMORY ONLY (CSV unchanged)")
        print(f"{'='*70}\n")
        return df_corrected
    else:
        print(f"\n✓ Phase Volume already in correct range (max={pv_max:.2f})")
        return df_chaos


def runitall():
    base = r'C:\Users\uceelmd\OneDrive\Uni\ERDS Study\Subjects\Data Collection'
    output_dir = os.path.join(base, 'Analysis Output')
    stat_dir = os.path.join(output_dir, 'Statistics')
    
    os.makedirs(stat_dir, exist_ok=True)
    subjects = sorted([d for d in os.listdir(base) if d.startswith('Subject') and os.path.isdir(os.path.join(base, d))])
    
    # ==================== LOAD EVERYTHING ONCE ====================
    print("\n" + "="*70)
    print("[STEP 1] LOADING ALL DATA (Single Pass)")
    print("="*70)
    
    df_linear_full = OrganisationalCalls.loadlin(base)  # One load!
    df_chaos_full = OrganisationalCalls.loadcha(base)   # One load!
    
    if df_linear_full is None or df_chaos_full is None:
        print("[ERROR] Failed to load data")
        exit()
    
    print(f"\nLoaded {len(df_linear_full):,} linear trials, {len(df_chaos_full):,} chaos trials")
    
    # Harmonise ONCE, reuse everywhere
    df_merged = OrganisationalCalls.harmonise(df_linear_full, df_chaos_full)
    if df_merged is None:
        print("[ERROR] Harmonisation failed")
        exit()
    
    print(f"Merged dataset: {len(df_merged):,} trials (after outlier removal)")

    # INSERT AFTER harmonise() CALL

    print(f"\n[DEBUG] BEFORE HARMOINISATION:")
    print(f"  Linear channels: {len(df_linear_full['Channel'].unique())} | Sample: {sorted(df_linear_full['Channel'].unique())[:10]}")
    print(f"  Chaos channels: {len(df_chaos_full['Channel'].unique())} | Sample: {sorted(df_chaos_full['Channel'].unique())[:10]}")

    common = set(df_linear_full['Channel'].unique()).intersection(set(df_chaos_full['Channel'].unique()))
    print(f"  Overlap channels: {len(common)}")
    print(f"  Missing from Linear: {len(set(df_linear_full['Channel'].unique()) - common)}")
    print(f"  Missing from Chaos: {len(set(df_chaos_full['Channel'].unique()) - common)}")

    print(f"\n[DEBUG] AFTER HARMONISATION:")
    print(f"  Merged channels: {sorted(df_merged['Channel'].unique())}")
    
    # Verify what we actually have
    n_subj = len(df_merged['Subject'].unique())
    n_chan = len(df_merged['Channel'].unique())
    n_cond = len(df_merged['Condition'].unique())
    n_trial_avg = df_merged.groupby(['Subject','Condition']).size().mean()
    
    print(f"Coverage: {n_subj} subjects × {n_chan} channels × {n_cond} conditions × ~{int(n_trial_avg)} trials")
    
    # ==================== CHAOS: PROCESS PER-SUBJECT ====================
    print("\n" + "="*70)
    print("[STEP 2] PER-SUBJECT CHAOS ANALYSIS")
    print("="*70)
    
    cha_results = []
    for subj in subjects:
        if subj == 'Subject21':  # Skip known problematic subject
            continue
            
        subj_cha_df = df_chaos_full[df_chaos_full['Subject'] == subj]
        if len(subj_cha_df) == 0:
            continue
        
        res = subjectwise.calcova(subj_cha_df, subj)
        if res:
            cha_results.append(res)
            subjectwise.save(res, stat_dir)
    
    print(f"\nProcessed {len(cha_results)} chaos subjects")
    
    # ==================== LINEAR: PARALLEL PROCESSING ====================
    print("\n" + "="*70)
    print("[STEP 3] PER-SUBJECT LINEAR ANALYSIS (New calcova_lin)")
    print("="*70)
    
    lin_results = []
    for subj in subjects:
        if subj == 'Subject21':
            continue
            
        subj_lin_df = df_linear_full[df_linear_full['Subject'] == subj]
        if len(subj_lin_df) == 0:
            continue
        
        res = LinearCalcova.calcova(subj_lin_df, subj)  # Mirror of chaos!
        if res:
            lin_results.append(res)
            subjectwise.save(res, stat_dir)  # Reuse same saver
    
    print(f"Processed {len(lin_results)} linear subjects")
    
    # ==================== CROSS-DOMAIN COMPARISON ====================
    print("\n" + "="*70)
    print("[STEP 4] CROSS-DOMAIN STABILITY TEST")
    print("="*70)
    
    comp_result = compare_domain_stabilities(cha_results, lin_results, stat_dir)
    
    if comp_result and comp_result['is_significant']:
        print(f"\n SIGNIFICANT RESULT: Chaos more stable p = {comp_result['p_diff']:.4f})")
    
    # ==================== CHANNEL-LEVEL EFFECT SIZES ====================
    print("\n" + "="*70)
    print("[STEP 5] CHANNEL-LEVEL EFFECT SIZES")
    print("="*70)
    
    # Use SAME merged dataframe (don't reload!)
    analyzer = ChannelLevelEffectSizeAnalysis(df_merged)
    analyzer.runana(stat_dir)
    
    # ==================== GROUP-LEVEL ANALYSES ====================
    print("\n" + "="*70)
    print("[STEP 6] GROUP-LEVEL ANALYSES")
    print("="*70)
    
    
    # Group correlations
    group_analyser = GroupCorrelationAnalysis(cha_results)
    group_analyser.runana(stat_dir)
    
    # Cross-condition
    cross_analyser = CrossConditionAnalysis(cha_results)
    cross_analyser.runana(stat_dir)
    
    # Multivariate
    mv_analyser = MultivariateFingerDiffAnalysis(cha_results)
    mv_analyser.runana(stat_dir)


    # ... existing code until after harmonise() ...

    # =========================================================================
    # FIX: Apply in-memory log transform BEFORE any plotting/analysis
    # =========================================================================
    print("\n[STEP 1c] Applying Phase Volume Safety Fix...")
    df_chaos_full = apply_in_memory_log_transform(df_chaos_full)


    print("\n[STEP 7] Generating Attractor Plots...")

    # Option A: Centroid-centered (RECOMMENDED for paper)
    plot_centroid_attractors(
        df_chaos=df_chaos_full,
        output_dir=os.path.join(stat_dir, 'Centroid_Attractors')
    )


    print("\n✓ All plots generated successfully!")
        
    
    # ==================== FINAL REPORT ====================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"  Files saved: {stat_dir}/")
    print(f"  Chaos subjects: {len(cha_results)}")
    print(f"  Linear subjects: {len(lin_results)}")
    if comp_result:
        print(f"  Domain stability: Chaos r={comp_result['r_chaos']:.4f}, Linear r={comp_result['r_linear']:.4f}")
        print(f"  Significance: {'YES' if comp_result['is_significant'] else 'NO'}")


if __name__ == '__main__':
    runitall()
