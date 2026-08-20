"""
Multivariate & Group-Level Analysis Module
Handles Hotelling's T², group correlations, and cross-condition effects
"""

import numpy as np
import os
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import pickle
from LvCrossCondition import domstab
from LvCSubject import Utils, OrganisationalCalls, ChannelLevelEffectSizeAnalysis, METRICS, PAIRS


class MultivariateFingerDiffAnalysis:

    def __init__(self, subject_results):
        self.results = subject_results
    
    def runana(self, outdir):
        print("\n" + "="*70)
        print("[INFO] 3D MULTIVARIATE FINGER DIFFERENTIATION")
        print("="*70)
        print("Method: Hotelling's T² (Pairwise) & Mahalanobis Distance")
        print("-"*70)

        analyse_dir = os.path.join(outdir, 'Multivariate_Finger_Diff')
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

        if all_pairs_data:
            df_agg = pd.DataFrame(all_pairs_data)
            
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
            
            df_agg_sorted = df_agg.sort_values(['Condition', 'Subject', 'Finger_A', 'Finger_B'])
            path_sum = os.path.join(analyse_dir, 'Group_Multivariate_Summary_FDR_Pairwise.csv')
            df_agg_sorted.to_csv(path_sum, index=False)
            
            print(f"\n[INFO] Aggregated results saved to {path_sum}")
            
            summary_rows = []
            for cond in conditions:
                sub_df = df_agg_sorted[df_agg_sorted['Condition'] == cond]
                if len(sub_df) == 0: continue
                
                total_tests = len(sub_df)
                sig_tests_raw = sub_df[sub_df['Is_Significant_Raw'] == True]['T2_Statistic'].count()
                sig_tests_fdr = sub_df[sub_df['Is_Significant_FDR'] == 'True']['T2_Statistic'].count()
                avg_md = sub_df['Mahalanobis_Distance'].mean()
                
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

    def runana(self, outdir):
        metrics = ['HFD', 'Lyapunov_Exponent', 'Phase_Volume']
        pairs = [('HFD', 'Lyapunov_Exponent'), ('HFD', 'Phase_Volume'), ('Lyapunov_Exponent', 'Phase_Volume')]
        pair_names = [f"{p[0]}-{p[1]}" for p in pairs]
        conditions = ['Random', 'Sequential']
        
        print("\n[INFO] Running Group-Level Correlation Analysis...")

        agg_data = self.aggz(conditions, pair_names)
        
        if not any(agg_data[c] for c in conditions):
            print("[ERROR] No sufficient data found for group analysis.")
            return None
        
        group_dir = os.path.join(outdir, 'Group_Correlation_Analysis')
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
            print(f"[INFO] Saved Group Pairwise Comparisons: {path}")

        print("\n[DONE] Group Correlation Analysis Complete.")
        return group_dir


class CrossConditionAnalysis:
    
    def __init__(self, subject_results):
        self.results = subject_results
        
    def normalitycheck(self, z_random, z_sequential):
        try:
            _, p_rand = stats.shapiro(z_random)
            _, p_seq = stats.shapiro(z_sequential)
            if p_rand > 0.05 and p_seq > 0.05:
                return 'ttest'
            else:
                return 'wilcoxon'
        except Exception:
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
    
    def runana(self, outdir):
        pair_names = ['HFD-Lyapunov_Exponent', 'HFD-Phase_Volume', 'Lyapunov_Exponent-Phase_Volume']
        
        print("\n" + "="*70)
        print("CROSS-CONDITION ANALYSIS: Sequential vs Random Effect Assessment")
        print("="*70)
        print("Goal: Identify FIXED BEHAVIOR (correlations stable across conditions)")
        print("-"*70)
        
        analysis_dir = os.path.join(outdir, 'Cross_Condition_Analysis')
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
                cohens_d = mean_diff / std_diff if std_diff > 0 else 0
                
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
        
        if df_condition_effects:
            df_out = pd.DataFrame(df_condition_effects)
            path = os.path.join(analysis_dir, 'Cross_Condition_Effects_Main.csv')
            df_out.to_csv(path, index=False)
            print(f"\n[INFO] Saved main results: {path}")
        
        if df_fixed_behavior_summary:
            df_out = pd.DataFrame(df_fixed_behavior_summary)
            path = os.path.join(analysis_dir, 'Fixed_Behavior_Summary.csv')
            df_out.to_csv(path, index=False)
            print(f"[INFO] Saved fixed behavior summary: {path}")
        
        if df_subject_details:
            df_out = pd.DataFrame(df_subject_details)
            path = os.path.join(analysis_dir, 'Subject_Level_Details.csv')
            df_out.to_csv(path, index=False)
            print(f"[INFO] Saved subject-level details: {path}")
        
        r_stab, p_corr = self.trianglecorn()
        
        with open(os.path.join(analysis_dir, 'Upper_Triangle_Matrix_Correlation.txt'), 'w') as f:
            f.write("Upper-Triangle Matrix Correlation (Chaos Domain)\n")
            f.write("=============================================\n\n")
            f.write(f"r_stability = {r_stab:.4f}\n") if r_stab else f.write("r_stability = N/A\n")
            f.write(f"P-value = {p_corr:.6f}\n") if p_corr else f.write("P-value = N/A\n")
            f.write(f"\nMethod: Pearson correlation between Fisher-Z vectors\n")
            f.write(f"extracted from upper-triangles of correlation matrices\n")
            f.write(f"(3 metric pairs x 5 fingers x N subjects per condition).\n")
        
        print(f"\n   Chaos Stability r = {r_stab:.4f} (p = {p_corr:.6f})")
        
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
            print("[ERROR] No results to summarise.")
        
        print(f"\n  Chaos Stability r = {r_stab:.4f} (p = {p_corr:.6f})" if r_stab else "\n  Chaos Stability r = N/A")
        print("\n[DONE] Cross-Condition Analysis Complete.")
        chr_stab = r_stab

        return analysis_dir, chr_stab


def stability(cha_results, df_merged, outdir):
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
    
    print("\n[LINEAR STABILITY] Computing using identical pipeline...")
    
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
    
    stab_dir = os.path.join(outdir, 'Unified_Stability_Script2_Method')
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
        print("  [ERROR] Could not complete comparison due to missing data")
    
    return cha_result, lin_result



def runitall():
    # ==================== HARDCODED PATHS ====================
    base = r'C:\Users\uceelmd\OneDrive\Uni\ERDS Study\Subjects\Data Collection'
    outdir = os.path.join(base, 'Analysis Output')
    stat_dir = os.path.join(outdir, 'Statistics')
    
    os.makedirs(stat_dir, exist_ok=True)
    print(f"[INFO] Output directory: {outdir}")
    print(f"[INFO] Statistics directory: {stat_dir}")
    
    subjects = sorted([d for d in os.listdir(base) 
                      if d.startswith('Subject') 
                      and os.path.isdir(os.path.join(base, d))])
    
    print(f"\n{'='*70}")
    print(f"MULTIVARIATE ANALYSIS PIPELINE")
    print(f"{'='*70}")
    print(f"[STEP 0] Found {len(subjects)} subjects")
    
    # ==================== OPTION 1: LOAD PRE-SAVED RESULTS ====================
    # If you already ran subject_analysis.py and saved pickle files
    cha_results_file = os.path.join(stat_dir, 'chaos_subject_results.pkl')
    lin_results_file = os.path.join(stat_dir, 'linear_subject_results.pkl')
    
    if os.path.exists(cha_results_file) and os.path.exists(lin_results_file):
        print("\n[INFO] Loading pre-computed subject results...")
        with open(cha_results_file, 'rb') as f:
            cha_results = pickle.load(f)
        with open(lin_results_file, 'rb') as f:
            lin_results = pickle.load(f)
        print(f"[INFO] Loaded {len(cha_results)} chaos subjects, {len(lin_results)} linear subjects")
    
    # ==================== OPTION 2: RE-RUN SUBJECT PROCESSING ====================
    else:
        print("\n[ERROR] Pre-computed results not found. Re-processing from raw data...")
        print("(Note: This duplicates work from subject_analysis.py)")
        
        # Load raw data
        df_linear_full = OrganisationalCalls.loadlin(base)
        df_chaos_full = OrganisationalCalls.loadcha(base)
        
        if df_linear_full is None or df_chaos_full is None:
            print("[ERROR] Failed to load data")
            exit()
        
        # Harmonise
        df_merged = OrganisationalCalls.harmonise(df_linear_full, df_chaos_full)
        if df_merged is None:
            print("[ERROR] Harmonisation failed")
            exit()
        
        print(f"Merged dataset: {len(df_merged):,} trials")
        
        # Process chaos subjects
        print("\n" + "="*70)
        print("Processing Chaos Subjects")
        print("="*70)
        cha_results = []
        for subj in subjects:
            if subj == 'Subject21':
                continue
            subj_cha_df = df_chaos_full[df_chaos_full['Subject'] == subj]
            if len(subj_cha_df) == 0:
                continue
            
            from LvCSubject import subjectwise
            res = subjectwise.calcova(subj_cha_df, subj)
            if res:
                cha_results.append(res)
                subjectwise.save(res, stat_dir)
        
        print(f"Processed {len(cha_results)} chaos subjects")
        
        # Process linear subjects
        print("\n" + "="*70)
        print("Processing Linear Subjects")
        print("="*70)
        lin_results = []
        for subj in subjects:
            if subj == 'Subject21':
                continue
            subj_lin_df = df_linear_full[df_linear_full['Subject'] == subj]
            if len(subj_lin_df) == 0:
                continue
            
            from LvCSubject import LinearCalcova
            res = LinearCalcova.calcova(subj_lin_df, subj)
            if res:
                lin_results.append(res)
                subjectwise.save(res, stat_dir)
        
        print(f"Processed {len(lin_results)} linear subjects")
        
        # Save for later use
        with open(cha_results_file, 'wb') as f:
            pickle.dump(cha_results, f)
        with open(lin_results_file, 'wb') as f:
            pickle.dump(lin_results, f)
        print(f"[INFO] Saved pre-computed results for future runs")
    
    # ==================== RUN MULTIVARIATE ANALYSES ====================
    print("\n" + "="*70)
    print("[STEP 1] MULTIVARIATE FINGER DIFFERENTIATION")
    print("="*70)
    mv_analyser = MultivariateFingerDiffAnalysis(cha_results)
    mv_dir = mv_analyser.runana(stat_dir)
    
    print("\n" + "="*70)
    print("[STEP 2] GROUP CORRELATION ANALYSIS")
    print("="*70)
    group_analyser = GroupCorrelationAnalysis(cha_results)
    group_dir = group_analyser.runana(stat_dir)
    
    print("\n" + "="*70)
    print("[STEP 3] CROSS-CONDITION ANALYSIS")
    print("="*70)
    cross_analyser = CrossConditionAnalysis(cha_results)
    cross_dir, stability_r = cross_analyser.runana(stat_dir)
    
    # ==================== CHANNEL LEVEL EFFECT SIZES ====================
    print("\n" + "="*70)
    print("[STEP 4] CHANNEL-LEVEL EFFECT SIZES")
    print("="*70)
    if 'df_merged' in locals():
        analyzer = ChannelLevelEffectSizeAnalysis(df_merged)
        analyzer.runana(stat_dir)
    else:
        print("[SKIP] Need merged dataframe (re-process with Option 2 above)")
    
    # ==================== DEBUG: INSPECT LINEAR STRUCTURE ====================
    print("\n" + "="*70)
    print("DEBUG: Examining Linear Results Structure")
    print("="*70)

    try:
        first_lin = lin_results[0]
        print("Top-level keys:", list(first_lin.keys()))
        
        fc = first_lin.get('finger_correlations', {})
        print("Finger correlations conditions:", list(fc.keys()))
        
        seq_fingers = fc.get('Sequential', {})
        print("Sequential fingers:", list(seq_fingers.keys()))
        
        if seq_fingers:
            first_finger = list(seq_fingers.keys())[0]
            finger_data = seq_fingers[first_finger]
            print(f"Finger {first_finger} keys:", list(finger_data.keys()))
            
            fisher_z = finger_data.get('fisher_z', {})
            print(f"Fishers Z pairs ({len(fisher_z)}):", list(fisher_z.keys())[:5])
            if fisher_z:
                print("Sample values:", dict(list(fisher_z.items())[:2]))
    except Exception as e:
        print(f"[ERROR] Debug failed: {e}")

    # ==================== CROSS-DOMAIN STABILITY COMPARISON ====================
    print("\n" + "="*70)
    print("[STEP 5] CROSS-DOMAIN STABILITY TEST")
    print("="*70)

    try:
        
        
        comp_result = domstab(cha_results, lin_results, stat_dir)
        
        if comp_result:
            print(f"\nCross-domain comparison complete:")
            print(f"  Chaos stability: r = {comp_result['r_chaos']:.4f}")
            print(f"  Linear stability: r = {comp_result['r_linear']:.4f}")
            print(f"  Difference: Δr = {comp_result['Δ_R']:+.4f}")
            print(f"  Significance: {'YES (p < 0.05)' if comp_result['is_significant'] else 'NO'}")
        else:
            print("[ERROR] Cross-domain comparison failed - insufficient data")
    except Exception as e:
        print(f"[ERROR] Could not run cross-domain comparison: {e}")
        import traceback
        traceback.print_exc()



    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"  Files saved to: {stat_dir}/")
    print(f"  Chaos subjects: {len(cha_results)}")
    print(f"  Linear subjects: {len(lin_results)}")
    print(f"  Stability r: {stability_r:.4f}" if stability_r else "  Stability r: N/A")
    print(f"  Sub-directories created:")
    print(f"    - {mv_dir}")
    print(f"    - {group_dir}")
    print(f"    - {cross_dir}")

if __name__ == '__main__':
    runitall()