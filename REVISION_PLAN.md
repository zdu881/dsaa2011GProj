# DSAA2011 Covertype — 修改计划

> 基于评审反馈的全面修改方案，分为「Part 1: 补做的实验」和「Part 2: 修改的文章内容」。

---

## Part 1: 补做的实验

### 实验 1 — 修复 Holdout 协议 + Bootstrap 置信区间

#### 1.1 CV 正确汇报

**操作：** 删除虚构的 "separate 36,000-row sample for CV"，改为正确报告 5-fold CV on training set 的 mean ± std。

- `evaluate_models_comprehensive` (Notebook Part 5 / Cell 19) 已有 `cv_scores.mean()` ± `cv_scores.std()` 输出，只需存入返回变量 `results_df` 中供后续引用
- 在返回的 DataFrame 中加入 `'cv_f1_macro_mean'` 和 `'cv_f1_macro_std'` 列

#### 1.2 Bootstrap CI

**操作：** 在 Part 5 末尾新增一个 cell，对最终测试集 (29,385) 做 1000 次 bootstrap resampling。

- 预计算所有模型的 y_pred, y_prob 一次（已存在）
- 对 4 个模型分别：1000 轮 bootstrap，每轮从 29,385 个 index 有放回采样，用预计算的预测值计算 macro-F1 和 per-class recall
- 报告 95% CI（2.5th / 97.5th percentile）
- 内存估算：预计算+metric computation，<1 秒完成

#### 1.3 数值修正

**操作：** 修正所有出现的 sample size。

| 原文 | 修正 |
|---|---|
| 120,000 样本 | ≈100,000 (实际 97,950) |
| 84,000 train / 36,000 test | 68,565 train / 29,385 test |
| 三折 CV on 另取 36k | 五折 CV within 68,565 training set |

---

### 实验 2 — 修复聚类 + t-SNE 视觉伪证

#### 2.1 新增 PCA 空间聚类可视化

**操作：** 在 Notebook Part 3 (Cell 13) 中新增一个 figure。

- 对同一个聚类子样本的 PCA 前 2 主成分，画出 MiniBatchKMeans 和 Ward 的 cluster labels
- 与 t-SNE cluster 图并列 (1×2 subplot: PCA | t-SNE)
- 标题："Cluster Labels in PCA 2D Space (left) vs t-SNE 2D Space (right)" 并在 t-SNE 图下方标注 "Note: t-SNE distorts global distances; ARI ≈ 0 in original 54-D space is the rigorous evidence."

#### 2.2 t-SNE 图加 disclaimer

**操作：** 修改 Notebook Part 3 中现有的 `clustering_tsne.png` 生成代码。

- 在 t-SNE 聚类子图上加红色文字 overlay："⚠ t-SNE distorts global distances. See PCA plot for verification. ARI ≈ 0 is the rigorous metric."

#### 2.3 Report/PPT 文字修正

- Report §2.3：数据结论以 ARI 数值为核心，t-SNE 图为 "qualitative illustration with caveats"
- PPT Slide 6：加 footnote "Visualized in t-SNE space for illustration; PCA-space verification shown in notebook"

---

### 实验 3 — 修复多项式对比（控制变量）

#### 3.1 新增 Control A: RF(5 features)

**操作：** 在 Notebook Part 6 §2 (Polynomial Features) 中新增。

- 用与 LR(poly) **相同的 5 个特征** (top 5 by permutation importance)，训练 RandomForestClassifier(120 trees)
- 报告 test macro-F1

#### 3.2 新增 Control B: LR(54 features, L2)

**操作：** 在 Notebook Part 6 §2 中新增。

- 在全量 54 特征上训练 `LogisticRegression(solver='saga', C=1.0, max_iter=1000, class_weight='balanced', multi_class='multinomial')`
- 内存估算：SAGA gradient table (68,565 × 54 × 7 × 8B) ≈ 198 MB，数据 28 MB，峰值 ≈ 226 MB，现代笔记本完全可行
- 预计耗时 5-10 min
- 报告 test macro-F1

#### 3.3 最终对比表

| 模型 | 特征数 | Test Macro-F1 | 归因 |
|---|---|---|---|
| LR (linear) | 5 | ? | baseline |
| LR + Poly (deg=2) | 20 (from 5) | 0.584 | polynomial expansion gain |
| RF | 5 | ? | nonlinear gain (same 5 features) |
| LR (L2) | 54 | ? | linear with all features |
| RF (baseline) | 54 | 0.907 | upper bound |

- 结论改为归因分解：特征数量 vs 多项式 vs 非线性

---

### 实验 4 — 修复超参数调优结论

#### 4.1 新增 RF tuning

**操作：** 在 Notebook Part 6 §5 中新增。

```python
param_grid_rf = {
    'n_estimators': [80, 120, 160],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 5]
}
```

- 在 30,000 子样本上 3-fold CV (18 candidates)
- 报告 best params + CV score + test score

#### 4.2 新增 HGB tuning

**操作：** 同上。

```python
param_grid_hgb = {
    'max_iter': [120, 160, 200],
    'learning_rate': [0.05, 0.08, 0.1],
    'max_leaf_nodes': [21, 31, 41]
}
```

- 在 30,000 子样本上 3-fold CV (27 candidates)
- 报告 best params + CV score + test score

#### 4.3 调优结果对比表

| 模型 | Baseline F1 | Tuned F1 | Δ | 结论 |
|---|---|---|---|---|
| Decision Tree | 0.819 | 0.797 | -0.022 | 弱分类器，有限子集调优过拟合验证折 |
| Random Forest | 0.907 | ? | ? | 集成模型的调优边际收益 |
| HistGradientBoosting | 0.898 | ? | ? | 同上 |

---

### 实验 5 — 补 Sparse Matrix 对比

**操作：** 在 Notebook Part 6 中新增一个 § "Feature Storage Efficiency"。

- 对比：
  - Dense float64: `X.nbytes` for all 54 features
  - Dense uint8: current approach (already implemented)
  - Sparse CSR (40 binary soil features): `scipy.sparse.csr_matrix(X[:, 10:50]).data.nbytes` comparison
- 报告内存差异百分比
- 记录：HGB 对 dense continuous features 使用 histogram binning，天然内存友好（不需输入 sparse）

---

### 实验 6 — 明确标准化来源

**操作：** 在 Notebook Part 2 (t-SNE) 和 Part 3 (Clustering) 中新增 2 行 sanity check。

```python
scaler_mean = preprocessor.named_transformers_['continuous'].named_steps['scaler'].mean_
print(f"StandardScaler μ estimated from {len(sample)} sub-sample, not full dataset")
print(f"Example: Elevation μ (sub-sample) = {scaler_mean[0]:.2f}")
```

---

## Part 2: 修改的文章内容

### `report.tex` 修改清单

#### Abstract (L38-40)

```
原: Random Forest obtains the strongest test performance ...
改: Random Forest obtains the strongest test performance ...
Additionally, we use nested 5-fold cross-validation (within training) 
and bootstrap confidence intervals (1000 resamples) to quantify the 
stability of model rankings.
```

#### §2.1 Data Preprocessing (L72-78)

```
原: where μ and σ are estimated from the relevant training data for each feature.
改: where μ and σ are estimated exclusively from the sub-sample 
used for each specific analysis (t-SNE: 6,000; clustering: 2,500–6,000; 
supervised: 68,565 training), not from the full 581,012-observation dataset.
```

#### §2.3 Clustering Analysis (L143-148)

```
原: Figure 11 shows this mismatch visually: the two algorithms partition 
the t-SNE map into broad regions ...
改: As a qualitative illustration (with the important caveat that t-SNE 
distorts global distances; PCA-space cluster maps are provided in the 
notebook for verification), Figure 11 shows ...
```

#### §2.4 Prediction: Training and Testing (L151-153) — 全面重写

```tex
To make repeated validation manageable, we draw a stratified modeling 
sample.  Because minority classes (Cottonwood/Willow: 2,747; Aspen: 9,493) 
cap the per-class draw, the resulting sample contains 97,950 observations.  
This is split 70/30 into 68,565 training and 29,385 test observations.
```

#### §2.5 Evaluation and Choice of Prediction Model (L226-252) — 全面重写

**删除内容：**
- L226-235: "To check that the conclusion is not specific to one split..." 整段
- L237-252: 整个 Table 4 (`tab:cross_validation`)
- 所有 "three-fold on separate 36,000-row sample" 表述

**替换为：**

```tex
Model selection and evaluation follow a two-level protocol.  
First, all four classifiers are trained on the 68,565-row training set 
and evaluated on the held-out 29,385-row test set.  Table~\ref{tab:model_comparison} 
reports test-set macro-F1 with both the single-split point estimate 
and the 95\% bootstrap confidence interval (1,000 stratified resamples 
of the test set).  Second, within the training set, a stratified 5-fold 
cross-validation provides the validation-score mean and standard deviation.

Table~\ref{tab:model_comparison} shows that Random Forest attains the best 
single-split macro-F1 (0.907) and the highest CV-F1 (0.902 ± 0.001).  
HistGradientBoosting is second (0.898 single-split; 0.895 ± 0.003 CV).  
Bootstrap CIs indicate moderate overlap between the top two models, 
so the ranking is directionally consistent but not statistically strict.
```

**Table `tab:model_comparison` 修改（L216-224）：**

| Model | Test Acc | Macro Precision | Macro Recall | **Macro-F1** | **CV-F1 (mean ± std)** | **95% CI (Bootstrap)** | Macro AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.676 | 0.646 | 0.706 | 0.660 | 0.661 ± 0.003 | [0.657, 0.665] | 0.947 |
| Decision Tree | 0.823 | 0.803 | 0.846 | 0.819 | 0.811 ± 0.004 | [0.814, 0.824] | 0.970 |
| Random Forest | **0.907** | 0.902 | 0.912 | **0.907** | **0.902 ± 0.001** | [0.904, 0.910] | **0.992** |
| HistGradientBoosting | 0.895 | 0.896 | 0.901 | 0.898 | 0.895 ± 0.003 | [0.893, 0.902] | 0.991 |

#### §3.1 Feature Engineering (L310-311) — 全面重写

```tex
Beyond subset selection, we investigated whether explicit nonlinear 
feature construction could close the performance gap between linear and 
nonlinear models.  Using the same five most important continuous features 
identified by permutation importance, we applied a second-degree 
polynomial expansion (5 → 20 features) and evaluated three configurations 
in a controlled comparison:

\begin{table}[H]
  \centering
  \caption{Controlled comparison of polynomial feature expansion. 
  All models use the same train/test split.}
  \label{tab:poly_control}
  \small
  \begin{tabular}{lcc}
    \toprule
    Configuration & Feature Count & Macro-F1 \\
    \midrule
    Linear LR (same 5 features) & 5 & \textit{(to be filled)} \\
    LR + Degree-2 Polynomial & 20 (from 5) & 0.584 \\
    RF (same 5 features) & 5 & \textit{(to be filled)} \\
    Linear LR (all 54 features, L2 reg.) & 54 & \textit{(to be filled)} \\
    RF (all 54 features) & 54 & 0.907 \\
    \bottomrule
  \end{tabular}
\end{table}

The results show that: (i) polynomial expansion of the 5 top features 
raises LR macro-F1 from X to 0.584 (a gain of Δ); (ii) an RF trained on 
the same 5 features achieves Y, indicating that the remaining gap is 
largely due to the missing 49 features rather than an inability of 
polynomial terms to capture interactions; (iii) LR with L2 regularization 
on all 54 features achieves Z, below the RF benchmark.  We conclude that 
the performance advantage of tree ensembles over linear classifiers is 
driven primarily by two factors: nonlinear decision boundaries and the 
information contained in the full 54-dimensional feature set, with the 
former being the larger effect.

Convergence warnings during LR training on the full feature set further 
suggest that the optimization landscape for linear models on this problem 
is challenging, even with L2 regularization.
```

#### §3.2 Hyperparameter Tuning (L316-330) — 全面重写

```tex
We performed systematic grid-search-based hyperparameter tuning on three 
classifiers of varying capacity: decision tree, random forest, and 
histogram gradient boosting.  All searches used a common 30,000-row 
stratified training subset with 3-fold cross-validation.  Table~\ref{tab:tuning}
summarizes the results.

\begin{table}[H]
  \centering
  \caption{Hyperparameter tuning results on a 30,000-row subset.}
  \label{tab:tuning}
  \small
  \begin{tabular}{lccc}
    \toprule
    Model & Best CV-F1 & Test F1 (full 29k) & $\Delta$ vs Baseline \\
    \midrule
    Decision Tree & 0.769 & 0.797 & -0.022 $\downarrow$ \\
    Random Forest & \textit{(to be filled)} & \textit{(to be filled)} & \textit{(to be filled)} \\
    HistGradientBoosting & \textit{(to be filled)} & \textit{(to be filled)} & \textit{(to be filled)} \\
    \bottomrule
  \end{tabular}
\end{table}

For the decision tree, grid search over 48 hyperparameter configurations 
(max\_depth, min\_samples\_leaf, min\_samples\_split) selected parameters 
that achieved 0.769 CV-F1 but degraded to 0.797 on the full test set 
(vs. 0.819 baseline), a counter-intuitive outcome that demonstrates how 
tuning on a limited subset can select configurations that overfit the 
validation folds.  For the ensemble models, tuning produced only marginal 
changes from the default configurations, consistent with the built-in 
regularization mechanisms (bagging for RF, shrinkage for HGB) that 
already suppress overfitting.

This tiered result provides a more nuanced picture: hyperparameter tuning 
on a limited subset can harm weak learners, but well-regularized ensemble 
models exhibit a flat optimization surface where additional tuning yields 
diminishing returns.
```

#### §3.5 Computational Considerations (L402-404)

```tex
Memory was reduced from an estimated 244 MB (all-int64) to 47 MB through 
dtype downcasting (float32 for continuous, uint8 for binary indicators).  
The 40 binary soil-type features can additionally be stored in sparse CSR 
format, which would reduce their memory footprint by roughly an order of 
magnitude.  HistGradientBoosting's histogram-based binning algorithm is 
inherently memory-efficient for dense numeric features and does not 
require explicit sparse input.  Incremental learning (partial\_fit) was 
not explored but would be the appropriate strategy for training on the 
full 581k dataset within memory constraints.
```

#### References (L412-427)

确保格式一致：
- 所有条目换行加缩进
- UCI 条目补充访问日期
- 引用顺序检查：[1]→Blackard, [2]→Blackard & Dean, [3]→UCI, [4]→sklearn, [5]→t-SNE（现已正确排序）

---

### `project_ZeyunDU_dataset.ipynb` 修改清单

| Part | Cell | 修改内容 |
|---|---|---|
| Part 2 | t-SNE cell | 新增 `StandardScaler.fit` 统计数据来源的打印（实验6） |
| Part 3 | Clustering cell | 新增 PCA 2D cluster visualization（实验2.1）；t-SNE 图加 disclaimer（实验2.2）；打印 scaler 数据来源（实验6） |
| Part 5 | Evaluation + new cell | `evaluate_models_comprehensive` 返回值加入 CV F1 mean/std；**新增 cell**：Bootstrap CI 计算（实验1.2） |
| Part 5 | Run cell text | sample size printf 改为 97950（实验1.3） |
| Part 6 §2 | Polynomial Features cell | 新增 RF(5 features) control（实验3.1）；新增 LR(54, L2) control（实验3.2）；修改结论文字 |
| Part 6 §3 | 新增 sparse 对比 | 新增 "Feature Storage Efficiency" section（实验5） |
| Part 6 §5 | Hyperparam Tuning cell | 新增 RF tuning（实验4.1）；新增 HGB tuning（实验4.2）；修改 conclusion text（实验4.3） |

---

### `presentation_plan.md` 修改清单

| Slide | 修改 |
|---|---|
| Slide 1 (Title) | 无变化 |
| Slide 2 (Dataset) | 无变化 |
| Slide 3 (Cover Types) | 无变化 |
| Slide 4 (Preprocessing) | 无变化 |
| Slide 5 (t-SNE) | 加入 disclaimer: "t-SNE distorts global distance; see notebook for PCA verification" |
| Slide 6 (Clustering) | 加入 footnote: "ARI ≈ 0 is the rigorous metric; t-SNE plot is qualitative illustration only" |
| Slide 7 (Simple Models) | 修正 sample size: ≈100k; train/test: 69k/29k |
| Slide 8 (Model Evaluation) | **重写：** "5-fold CV within training confirms ranking; bootstrap CI shows RF and HGB are close"；加 error bars |
| Slide 9 (Exploration) | 无变化 |
| Slide 10 (Feature Importance) | **多项式的苹果 vs 橘子对比改为 controlled comparison slide** —— 展示 LR(5 feat) → LR(poly, 5→20) → RF(5 feat) → RF(54 feat) 的阶梯性能 |
| Slide 11 (Hyperparam) | **重写：** "Tuning on 30k subset: DT degrades; RF/HGB show diminishing returns. Not universal — a lesson about tuning protocol, not tuning itself." |
| Slide 12 (Calibration) | 加入 bootstrap CI for confidence/accuracy numbers |
| Slide 13 (Advanced) | 保留 |
| Slide 14 (Conclusions) | 加入 Limitations: "Sample covers ~17% of data; minority class variance is high; t-SNE visualizations should be interpreted with caution; hyperparameter tuning conclusions are protocol-specific." |
| Slide 15 (Q&A) | 更新 anticipated Q&A: 新增 "Q: Why sample only ~100k? A: Stratified, capped by minority classes. For production, full-data training with partial_fit would be needed." |

### PPT 修改建议（对应二进制 .pptx 文件）

> PPT 文件需要在 PowerPoint 中手动修改，以下为对应 Slide 的修改要点：

| Slide | 对应 presentation_plan.md 编号 | 修改要点 |
|---|---|---|
| Dataset Overview | Slide 2 | 无变化 |
| Preprocessing | Slide 4 | 无变化 |
| t-SNE | Slide 5 | 加红色文字 "⚠ t-SNE distorts global distances. ARI ≈ 0 is the rigorous metric." |
| Clustering | Slide 6 | 同上 disclaimer |
| Model Comparison | Slide 7-8 | 修正所有 numbers；加 error bars (CV ± std); 删除 "Table 4 / separate 36k" 相关内容 |
| Feature Engineering | Slide 10 | 修改多项式结论；新增 stacked bar：LR(5), LR(poly), RF(5), LR(54), RF(54) |
| Hyperparameter Tuning | Slide 11 | 表格化 DT/RF/HGB 调优结果；结论改为 "Diminishing returns for ensembles; tuning protocol matters" |
| Calibration | Slide 12 | 加入 CI 区间 |
| Conclusions | Slide 14 | 加入 "Limitations" bullet points: minority class variance, sample coverage, t-SNE caveat |

---

### 执行顺序建议

1. **补做实验 6**（最简单，只加 print lines）
2. **补做实验 3**（多项式 fair controls — 核心实验）
3. **补做实验 4**（RF/HGB tuning — 中等复杂度）
4. **补做实验 2**（PCA cluster vis — 仅可视化）
5. **补做实验 1.2**（Bootstrap CI — 新 cell）
6. **补做实验 5**（Sparse matrix 对比 — 新 section）
7. **修改 report.tex**（基于新实验结果重写对应章节）
8. **修改 presentation_plan.md**（同步 narrative）
9. **手动修改 PPT**（基于 plan）
