import numpy as np
import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from LvCSubject import Utils, METRICS


def domstab(cha_results, lin_results, outdir):
    """
    Compare stability between chaos and linear domains
    FIX: Properly initialize counters and extract all Z-values
    """
    # ==========================================
    # EXTRACT ALL FISHER-Z VALUES FROM EACH DOMAIN
    # ==========================================
    
    # CHAOS: Collect ALL Z-values (regardless of pair name)
    cha_seq_all = []
    cha_rand_all = []
    n_chaos_valid = 0
    
    for subj_idx, cha_res in enumerate(cha_results):
        subj_id = cha_res.get('subject', f'Subj{subj_idx}')
        cha_rand = cha_res.get('finger_correlations', {}).get('Random', {})
        cha_seq = cha_res.get('finger_correlations', {}).get('Sequential', {})
        
        subj_has_data = False
        
        for cls in cha_rand.keys():
            for pair, z_rand in cha_rand[cls].get('fisher_z', {}).items():
                z_seq = cha_seq.get(cls, {}).get('fisher_z', {}).get(pair)
                
                if z_rand is not None and z_seq is not None:
                    cha_rand_all.append({'subj': subj_id, 'cls': cls, 'pair': pair, 'z': z_rand})
                    cha_seq_all.append({'subj': subj_id, 'cls': cls, 'pair': pair, 'z': z_seq})
                    subj_has_data = True
        
        if subj_has_data:
            n_chaos_valid += 1
    
    # LINEAR: Same extraction logic
    lin_seq_all = []
    lin_rand_all = []
    n_lin_valid = 0
    
    for subj_idx, lin_res in enumerate(lin_results):
        subj_id = lin_res.get('subject', f'Subj{subj_idx}')
        lin_rand = lin_res.get('finger_correlations', {}).get('Random', {})
        lin_seq = lin_res.get('finger_correlations', {}).get('Sequential', {})
        
        subj_has_data = False
        
        for cls in lin_rand.keys():
            for pair, z_rand in lin_rand[cls].get('fisher_z', {}).items():
                z_seq = lin_seq.get(cls, {}).get('fisher_z', {}).get(pair)
                
                if z_rand is not None and z_seq is not None:
                    lin_rand_all.append({'subj': subj_id, 'cls': cls, 'pair': pair, 'z': z_rand})
                    lin_seq_all.append({'subj': subj_id, 'cls': cls, 'pair': pair, 'z': z_seq})
                    subj_has_data = True
        
        if subj_has_data:
            n_lin_valid += 1
    
    print(f"\nExtracted {len(cha_seq_all)} chaos Z-values")
    print(f"         {len(lin_seq_all)} linear Z-values")
    print(f"Across {n_chaos_valid} chaos subjects, {n_lin_valid} linear subjects")
    
    # ==========================================
    # COMPUTE STABILITY WITHIN EACH DOMAIN
    # ==========================================
    
    cha_seq_vecs = [d['z'] for d in cha_seq_all]
    cha_rand_vecs = [d['z'] for d in cha_rand_all]
    lin_seq_vecs = [d['z'] for d in lin_seq_all]
    lin_rand_vecs = [d['z'] for d in lin_rand_all]
    
    if len(cha_seq_vecs) < 5 or len(lin_seq_vecs) < 5:
        print("[ERROR] Insufficient data for comparison")
        return None
    
    r_chaos, p_chaos = stats.pearsonr(cha_seq_vecs, cha_rand_vecs)
    r_linear, p_linear = stats.pearsonr(lin_seq_vecs, lin_rand_vecs)
    
    print(f"\nDomain Stability Results:")
    print(f"  Chaos:   r = {r_chaos:.4f}, p = {p_chaos:.2e}")
    print(f"  Linear:  r = {r_linear:.4f}, p = {p_linear:.2e}")
    print(f"  Difference: Δr = {r_chaos - r_linear:+.4f}")
    
    # === TEST IF DIFFERENCE IS SIGNIFICANT USING FISHER-R-TO-Z ===
    z_chaos = Utils.fisher_z(r_chaos)
    z_linear = Utils.fisher_z(r_linear)
    
    se_diff = np.sqrt(1/(len(cha_seq_vecs) - 3) + 1/(len(lin_seq_vecs) - 3))
    z_stat_diff = abs(z_chaos - z_linear) / se_diff
    p_diff = 2 * (1 - stats.norm.cdf(z_stat_diff))
    
    ci_low = (z_chaos - z_linear) - 1.96 * se_diff
    ci_high = (z_chaos - z_linear) + 1.96 * se_diff
    
    # Back-transform CI to correlation scale for plotting
    diff_r_back = Utils.fisher_z_inverse(z_chaos - z_linear)
    ci_low_r = Utils.fisher_z_inverse((z_chaos - z_linear) - 1.96 * se_diff)
    ci_high_r = Utils.fisher_z_inverse((z_chaos - z_linear) + 1.96 * se_diff)
    
    print(f"\nSignificance Test:")
    print(f"  Z-statistic (difference): {z_stat_diff:.3f}")
    print(f"  p-value: {p_diff:.4f}")
    print(f"  95% CI for Δr (Fisher-Z scale): [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Is Chaos more stable? {'YES' if p_diff < 0.05 and r_chaos > r_linear else 'NO'}")

    
    # ========================================
    # FIGURE 1: Bar Plot (matching your style)
    # ========================================
    fig1 = plt.figure(figsize=(10, 7))
    ax1 = fig1.add_subplot(111)
    
    domains = ['Linear\n(Spectral)', 'Chaos\n(Non-linear)']
    r_values = [r_linear, r_chaos]
    colors = ['#4a9bbf', '#e86a5a']  # Blue for Linear, Salmon/Red for Chaos
    
    # Create bars with thick black borders
    bars = ax1.bar(domains, r_values, color=colors, edgecolor='black', linewidth=2.5, width=0.6)
    

    # Red dashed threshold line at 0.8
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    positions = np.arange(len(domains))
        
    bars = ax1.bar(positions, r_values, width=0.5, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=1.2)
        
    for i, (bar, r) in enumerate(zip(bars, r_values)):
            if r is not None:
                ax1.text(bar.get_x() + bar.get_width() / 2, r + 0.015,
                        f'r = {r:.3f}', ha='center', va='bottom',
                        fontsize=12, fontweight='bold')
        
    ax1.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label='Fixed Behaviour Threshold (r > 0.8)')
        
    ax1.set_xticks(positions)
    ax1.set_xticklabels(domains, fontsize=12)
    ax1.set_ylabel('Stability (r)', fontsize=13)
    ax1.set_title('Cross-Condition Stability Domain Comparison',
                     fontsize=14, pad=15)
    ax1.set_ylim(min(r_values) - 0.1, 1.0)
    ax1.legend(loc='upper right', fontsize=10)
        
    fig1.tight_layout()
    save_path = os.path.join(outdir, 'Figure1_Domain_Stability.png')
    fig1.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [INFO] Saved: {save_path}")


    
    # Save CSV results
    results_df = pd.DataFrame([{
        'Domain': 'Chaos',
        'R_Stability': round(r_chaos, 4),
        'P_Value': f"{p_chaos:.2e}",
        'N_Vector_Elements': len(cha_seq_vecs),
        'Meets_Threshold_0.8': 'Yes' if r_chaos >= 0.8 else 'No'
    }, {
        'Domain': 'Linear',
        'R_Stability': round(r_linear, 4),
        'P_Value': f"{p_linear:.2e}",
        'N_Vector_Elements': len(lin_seq_vecs),
        'Meets_Threshold_0.8': 'Yes' if r_linear >= 0.8 else 'No'
    }])
    
    comparison_df = pd.DataFrame([{
        'Comparison': 'Chaos vs Linear',
        'Δ_R_Stability': round(r_chaos - r_linear, 4),
        'Fisher_Z_Difference': round(z_chaos - z_linear, 4),
        'SE_Difference': round(se_diff, 4),
        'Z_Statistic': round(z_stat_diff, 3),
        'P_Value_Diff': round(p_diff, 6),
        'CI_95_Low': round(ci_low_r, 4),
        'CI_95_High': round(ci_high_r, 4),
        'Significant_at_p0.05': 'Yes' if p_diff < 0.05 else 'No',
    }])
    
    results_df.to_csv(os.path.join(outdir, 'Domain_Stability_Results.csv'), index=False)
    comparison_df.to_csv(os.path.join(outdir, 'Domain_Comparison_Test.csv'), index=False)
    
    print(f"[INFO] Saved results CSVs")
    
    return {
        'r_chaos': r_chaos,
        'r_linear': r_linear,
        'p_diff': p_diff,
        'is_significant': p_diff < 0.05,
        'ci_low': ci_low_r,
        'ci_high': ci_high_r,
        'results_df': results_df,
        'comparison_df': comparison_df,
        'Δ_R': r_chaos - r_linear
    }


def runitall(cha_subject_results, lin_subject_results, outdir):

    print("\n" + "="*70)
    print("CROSS-DOMAIN STABILITY ANALYSIS")
    print("="*70)
    print(f"Chaos subjects: {len(cha_subject_results)}")
    print(f"Linear subjects: {len(lin_subject_results)}")
    print("-"*70)
    
    result = domstab(cha_subject_results, lin_subject_results, outdir)
    
    if result:

        print(f"[INFO] ANALYSIS COMPLETE")
        print(f"  Chaos stability:   r = {result['r_chaos']:.4f}")
        print(f"  Linear stability:  r = {result['r_linear']:.4f}")
        print(f"  Significance:      {'YES (p < 0.05)' if result['is_significant'] else 'NO'}")
        print(f"  95% CI for Δr:     [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
    else:
        print("\n✗ ANALYSIS FAILED - insufficient data")
    
    return result


