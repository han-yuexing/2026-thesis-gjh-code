import numpy as np
from scipy import stats

def perform_significance_test(results1, results2, metric_name="Accuracy", logger=None):
    """
    对两组结果进行显著性检验
    使用配对t检验和Wilcoxon符号秩检验
    
    参数:
        results1: 第一组结果（列表或数组）
        results2: 第二组结果（列表或数组）
        metric_name: 指标名称
        logger: 日志记录器（可选）
    
    返回:
        包含检验统计量和p值的字典
    """
    # 确保两组结果长度相同（配对检验要求）
    if len(results1) != len(results2):
        print(f"警告: 两组结果长度不同 ({len(results1)} vs {len(results2)})")
        print("将使用独立样本t检验和Mann-Whitney U检验")
        
        # 独立样本t检验
        t_statistic, t_pvalue = stats.ttest_ind(results1, results2)
        
        # Mann-Whitney U检验 (非参数检验)
        u_statistic, u_pvalue = stats.mannwhitneyu(results1, results2)
        
        print(f"\n=== {metric_name} 显著性检验结果 ===")
        print(f"独立样本t检验: t-statistic = {t_statistic:.4f}, p-value = {t_pvalue:.4f}")
        print(f"Mann-Whitney U检验: U-statistic = {u_statistic:.4f}, p-value = {u_pvalue:.4f}")
        
        if t_pvalue < 0.05:
            print(f"独立样本t检验结果: 差异显著 (p < 0.05)")
        else:
            print(f"独立样本t检验结果: 差异不显著 (p >= 0.05)")
        
        if u_pvalue < 0.05:
            print(f"Mann-Whitney U检验结果: 差异显著 (p < 0.05)")
        else:
            print(f"Mann-Whitney U检验结果: 差异不显著 (p >= 0.05)")
        
        # 计算效应量 (Cohen's d)
        mean1, mean2 = np.mean(results1), np.mean(results2)
        std1, std2 = np.std(results1, ddof=1), np.std(results2, ddof=1)
        pooled_std = np.sqrt(((len(results1)-1)*std1**2 + (len(results2)-1)*std2**2) / (len(results1)+len(results2)-2))
        cohens_d = (mean1 - mean2) / pooled_std
        print(f"Cohen's d (效应量): {cohens_d:.4f}")
        
        if logger:
            logger.info(f"=== {metric_name} 显著性检验结果 ===")
            logger.info(f"独立样本t检验: t-statistic = {t_statistic:.4f}, p-value = {t_pvalue:.4f}")
            logger.info(f"Mann-Whitney U检验: U-statistic = {u_statistic:.4f}, p-value = {u_pvalue:.4f}")
            logger.info(f"Cohen's d: {cohens_d:.4f}")
        
        return {
            't_statistic': t_statistic,
            't_pvalue': t_pvalue,
            'u_statistic': u_statistic,
            'u_pvalue': u_pvalue,
            'cohens_d': cohens_d
        }
    
    else:
        # 配对t检验
        t_statistic, t_pvalue = stats.ttest_rel(results1, results2)
        
        # Wilcoxon符号秩检验 (非参数检验)
        wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(results1, results2)
        
        print(f"\n=== {metric_name} 显著性检验结果 ===")
        print(f"配对t检验: t-statistic = {t_statistic:.4f}, p-value = {t_pvalue:.4f}")
        print(f"Wilcoxon符号秩检验: statistic = {wilcoxon_statistic:.4f}, p-value = {wilcoxon_pvalue:.4f}")
        
        if t_pvalue < 0.05:
            print(f"配对t检验结果: 差异显著 (p < 0.05)")
        else:
            print(f"配对t检验结果: 差异不显著 (p >= 0.05)")
        
        if wilcoxon_pvalue < 0.05:
            print(f"Wilcoxon检验结果: 差异显著 (p < 0.05)")
        else:
            print(f"Wilcoxon检验结果: 差异不显著 (p >= 0.05)")
        
        # 计算配对效应量
        differences = np.array(results1) - np.array(results2)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff
        print(f"Cohen's d (效应量): {cohens_d:.4f}")
        
        if logger:
            logger.info(f"=== {metric_name} 显著性检验结果 ===")
            logger.info(f"配对t检验: t-statistic = {t_statistic:.4f}, p-value = {t_pvalue:.4f}")
            logger.info(f"Wilcoxon符号秩检验: statistic = {wilcoxon_statistic:.4f}, p-value = {wilcoxon_pvalue:.4f}")
            logger.info(f"Cohen's d: {cohens_d:.4f}")
        
        return {
            't_statistic': t_statistic,
            't_pvalue': t_pvalue,
            'wilcoxon_statistic': wilcoxon_statistic,
            'wilcoxon_pvalue': wilcoxon_pvalue,
            'cohens_d': cohens_d
        }


if __name__ == "__main__":
    # 假设有两组实验结果
    config1_results = [0.7333, 0.8667, 0.8667, 0.8667, 0.8000, 0.7667, 0.8667, 0.8333, 0.7667, 0.9000]
    config2_results = [0.8077, 0.8462, 0.8846, 0.7692, 0.9231, 0.8846, 0.8077, 0.9231, 0.8846, 0.7308]

    perform_significance_test(config1_results, config2_results, "Validation Accuracy")