# PPT 修改提案

> 目标文件：`Forest Cover Type Analysis - DSAA2011 Project(2).pptx`
> 当前状态：所有数值已修正，以下为结构性 / 文本级改动。

---

## Slide 6 — Clustering Results

### 现状
- 标题：`MiniBatchKMeans (K=2, left) and Hierarchical Clustering (K=2, right)`
- 正文第一段：`MiniBatchKMeans (K=2) Identifies Natural Groupings`，但继续说 `partitions the data into three distinct clusters`（自相矛盾）
- 正文第二段：`Agglomerative Ward (K=2) Reveals Super-Clusters`
- 缺 ARI 数值

### 应改为

**标题 / 正文 K 值**：

| 位置 | 当前值 | 改为 |
|---|---|---|
| 主标题 | K=2 / K=2 | **K=3 / K=4**（MiniBatchKMeans 最佳 K=3, Ward 最佳 K=4 per silhouette） |
| 正文段 1 标题 | MiniBatchKMeans (K=2) | MiniBatchKMeans (K=3, sil=0.182) |
| 正文段 1 内容 | "three distinct clusters" | 保留（K=3 后不再矛盾） |
| 正文段 2 标题 | Hierarchical (K=2) | Ward Hierarchical (K=4, sil=0.156) |
| 正文段 4 | "Limitations for Fine-Grained Classification" | 末尾加："ARI=0.075 (K-Means) and 0.061 (Ward) vs. true labels — far below supervised models (0.908)." |

**图注 footnote**（在 t-SNE 聚类图下方或正文）：  
`"⚠ t-SNE distorts global distances. ARI computed in original 54-D space is the rigorous metric. PCA-space verification plots provided in notebook."`

---

## Slide 8 — Evaluation and Choice of Prediction Model

### 现状
- 两张表只有 LR 和 DT 两行，缺失 RF 和 HGB
- 文字 "Random Forest is the recommended model" 后面紧接 "Random Forest achieves..."（无空格拼接）
- 无 RF / HGB 的具体数字支撑推荐

### 应改为

**表 1（Overfitting Analysis）—— 加两行**：

| Model | Train F1 | Test F1 | Gap | Status |
|---|---|---|---|---|
| Logistic Regression | 0.6620 | 0.6597 | 0.0023 | No overfitting |
| Decision Tree | 0.8649 | 0.8194 | 0.0455 | Mild overfitting |
| **Random Forest** | **0.9864** | **0.9076** | **0.0788** | **Mild overfitting** |
| **HistGradientBoosting** | **0.9449** | **0.8982** | **0.0467** | **Mild overfitting** |

**表 2（Test Set Metrics）—— 加四行（RF + HGB train/test）**：

在原表 Decision Tree Test 行下面追加：

| Model | Dataset | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | CV F1 (Macro) | Macro AUC |
|---|---|---|---|---|---|---|---|
| Random Forest | Train | 0.9884 | 0.9822 | 0.9905 | 0.9864 | — | — |
| | Test | **0.9076** | 0.9025 | 0.9134 | **0.9076** | 0.9019±0.0009 | 0.9923 |
| HistGradientBoosting | Train | 0.9378 | 0.9433 | 0.9469 | 0.9449 | — | — |
| | Test | 0.8951 | 0.8957 | 0.9014 | 0.8982 | 0.8946±0.0030 | 0.9907 |

**文字描述**（替换 "Random Forest is the recommended modelRandom Forest achieves..."）：  

> Random Forest is the recommended model (F1=0.908, AUC=0.992). Bootstrap 95% CIs do not overlap (RF: [0.904, 0.911] vs HGB: [0.894, 0.902]), and a McNemar test confirms RF is significantly better (χ²=41.5, p<0.001).

---

## Slide 11 — Hyperparameter Tuning

### 现状
- 残留文本片段 "trade-off. Hyperparameter tun" 等拼接头尾
- "The origin" 行尾截断

### 应改为（精简文本）

**第一段**（替换当前 "Tuning results: Grid search..." 整段）：

> Grid search on 30k subset (48 DT configs, 3-fold CV): best CV-F1=0.769 but test F1=0.797 (baseline 0.819).

**第二段**（替换当前 "The tuned best config..." 整段）：

> Scaling to ensemble models: RF 0.908→0.886, HGB 0.898→0.889. All three models degraded — the 30k subset is too small for reliable tuning regardless of model class.

**第三段**（保留但修复结尾）：

> Increasing RF trees from 120→200 gave only +0.001 F1 gain. Defaults were near-optimal. Modern AutoML frameworks (Optuna, Hyperopt) with Bayesian pruning may outperform simple grid search on limited subsets.

---

## Slide 12 — Model Calibration Analysis

### 现状
- 左侧大数字 "0.128" 是 ECE 旧值
- 缺 HGB 校准对比
- 右侧文字 "RF mean conf=0.810" 但缺口径说明

### 应改为

**左侧数字**：

| 标签 | 当前 | 改为 |
|---|---|---|
| RF ECE（大号数字） | 0.128 | **0.125** |
| 新增 HGB ECE 指标 | （无） | 卡片 + 数字："HGB ECE = 0.041 (best calibrated)" |

**右侧文字**（替换当前 "Methodology Impact: Expected Calibration Error..." 整段）：

> **Methodology Impact**: RF shows systematic under-confidence (ECE=0.125, mean top-label conf=0.783 vs accuracy=0.908). HGB is well-calibrated (ECE=0.041). Platt (sigmoid) scaling on RF reduces Brier from 0.024→0.021. Paired McNemar test: χ²=41.5, p<0.001, confirming RF > HGB despite close absolute scores.

**底部文字**（保留下半段，修数字）：

> Performance varies by elevation band: accuracy 0.863→0.958. Cache la Poudre is hardest (F1=0.716). Error predictions: mean confidence 0.582 vs 0.810 for correct predictions.

---

## Slide 14 — Conclusion

### 现状
- "Limitations:" 段是单行长段落，可读性差

### 应改为

**"Comprehensive Evaluation with Metrics" block 末尾的 Limitations 段**，拆为 bullet：

> **Limitations**  
> • Sample covers ~17% of full dataset (97,950 / 581,012)  
> • Bootstrap CIs do NOT overlap; McNemar p<0.001 confirms RF > HGB  
> • Full-data SOTA (gradient boosting): 0.95-0.96 accuracy — a gap remains  
> • 30k-subset tuning degrades all model classes; AutoML not explored  
> • t-SNE distorts global distances; PCA verification provided

---

## 执行说明

| 改动类型 | 方式 | 涉及 Slide |
|---|---|---|
| 文本替换 | python-pptx `run.text.replace()` | 6, 12 |
| 表格加行 | python-pptx `table.rows.add_row()` + 填充单元格 | 8 |
| 新增形状（HGB ECE 卡片） | PowerPoint 手动加 | 12 |
| Bullet 文本重排 | PowerPoint 手动拆行 | 14 |

Slide 8 表格加行和 Slide 12 新增指标卡片需要手动在 PowerPoint 中操作（python-pptx 无法可靠地复制单元格格式），其余可脚本化完成。
