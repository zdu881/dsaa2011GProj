# DSAA2011 Covertype Presentation Plan

## Time Allocation: 15 minutes total (12 min talk + 3 min Q&A)

## Key Metrics to Memorize

| Metric | Value |
|---|---|
| Dataset size | 581,012 rows × 54 features |
| Modeling sample | 97,950 (train: 68,565 / test: 29,385) |
| Classes | 7 forest cover types (severe imbalance: 85.3% in top 2) |
| Best Model | Random Forest |
| Test Accuracy | 0.908 |
| Macro F1 | 0.908 (95% CI: [0.904, 0.911]) |
| Macro OVR AUC | 0.992 |
| Clustering ARI vs labels | ≤0.12 (Euclidean or Gower — clusters ≠ cover types) |
| Dominant feature | Elevation (RF impurity = 0.243) |

---

## Slide-by-Slide Script

### Slide 1 — Title (30 sec)

**Speaker:** Good morning/afternoon. Our project analyzes the UCI Forest Covertype dataset
— 581,000 observations, 54 cartographic features, and a 7-class prediction task.
We'll walk through preprocessing, clustering, model selection, and open-ended analysis.

### Slide 2 — Dataset Overview (60 sec)

**Speaker:** The data comes from Roosevelt National Forest in Colorado. Each row is a
30×30 meter patch with terrain measurements — elevation, slope, aspect, distances to
roads and water, plus wilderness area and soil type indicators. The key challenge:
classes 1 and 2 — Spruce/Fir and Lodgepole Pine — dominate with over 85% of the data.
Minority classes like Cottonwood/Willow make up only 0.5%. So accuracy alone would be
misleading; we use macro-averaged metrics throughout.

*(Gesture at class distribution chart on slide)*

### Slide 3 — Forest Cover Types (45 sec)

**Speaker:** Quick visual gallery of the seven classes — from high-elevation Spruce/Fir
and stunted Krummholz at the alpine treeline, to Ponderosa Pine in warm dry zones and
riparian Cottonwood. Each has distinct ecological requirements tied to elevation and
moisture, which is why terrain features turn out to be strong predictors.

### Slide 4 — Data Preprocessing (60 sec)

**Speaker:** Preprocessing is straightforward. No missing values — 100% complete.
Continuous variables are standardized with Z-score normalization for linear and
distance-based methods. Binary indicators stay as-is. We apply dtype downcasting:
float32 for continuous, uint8 for indicators. This reduces memory from 244 MB to 47 MB
— an 81% reduction — which makes repeated analysis feasible on a laptop. All splits
are stratified to preserve rare class proportions.

### Slide 5 — t-SNE Visualization (75 sec)

**Speaker:** We run t-SNE on a stratified 6,000-row sample after PCA to 30 dimensions.
Key observations: classes 1 and 2 form the biggest clusters but overlap heavily.
Class 4 — Cottonwood/Willow — is somewhat isolated, confirming its unique ecological
niche. Classes 5 and 6 show significant overlap. The takeaway: there is local structure,
but the seven labels are not clean global islands. This suggests the need for nonlinear
models and contextual features beyond a single 2D projection.

### Slide 6 — Clustering Analysis (60 sec)

**Speaker:** We test MiniBatchKMeans and Agglomerative Ward clustering.
MiniBatchKMeans peaks at K=3 (silhouette 0.182), Ward at K=4 (silhouette 0.156).
The crucial number: adjusted Rand index against the true labels is only 0.075 (K-Means)
and 0.103 (Ward) — far below any supervised model. Why such a large gap? Two structural
reasons: (i) the dataset mixes 10 continuous terrain variables with 44 binary indicators,
so Euclidean distance — the metric K-Means and Ward optimize — poorly captures
similarity in high-dimensional mixed-type space; (ii) clustering minimizes internal
variance, while the labels are defined by ecological criteria that don't align with
variance-minimizing partitions. To eliminate the distance as a confound, we also tested
Gower distance + Ward on the same subset. The result: ARI only 0.124 — still useless.
The bottleneck is not the distance metric; it's the clustering objective itself. Tree-based
models, by contrast, use axis-aligned splits that naturally handle mixed feature types.
(Cluster visualizations are shown in both t-SNE and PCA space; t-SNE distorts global
distances, so ARI in original 54-D space is the rigorous evidence.) So for prediction,
we must use supervised methods.

### Slide 7 — Simple Prediction Models (90 sec)

**Speaker:** We first train two baseline classifiers. Logistic regression — a linear
baseline — gets test accuracy 0.676 and macro-F1 only 0.660. It reaches macro AUC 0.947,
so the probability ranking is useful but class decisions are weak. The decision tree,
with nonlinear splits, raises accuracy to 0.823 and macro-F1 to 0.819. However, the
train-test gap (0.865 vs 0.819) shows moderate overfitting. These baselines tell us:
the problem requires ensembles that reduce variance while preserving nonlinear capability.

### Slide 8 — Model Evaluation and Choice (90 sec)

**Speaker:** We extend to Random Forest and HistGradientBoosting. Random Forest wins
across the board: accuracy 0.908, macro-F1 0.908, macro AUC 0.992. HistGradientBoosting
follows at 0.898 macro-F1 with the highest macro precision (0.896).  5-fold CV within the
training set confirms the same ranking (RF: 0.902±0.001, HGB: 0.895±0.003).
Bootstrap 95% CIs do not overlap (RF: [0.904, 0.911], HGB: [0.894, 0.902]),
and a paired McNemar test gives $\chi^2=41.5$, $p<0.001$ — confirming that RF and HGB
differ systematically in their per-sample error patterns. However, the macro-F1 gap is
only 0.010, a practically negligible difference. Choose based on deployment needs:
calibration (HGB ECE=0.041) or training speed (RF ~2 sec).

### Slide 9 — Open-Ended Exploration Overview (45 sec)

**Speaker:** Beyond baseline models, we pursue six exploration directions: feature
importance, feature engineering with controlled comparison, hyperparameter tuning across
three model classes, calibration, error analysis, and model interpretability. These
collectively validate that topographic factors — especially elevation — drive predictions,
while soil and wilderness context provide complementary signal.

### Slide 10 — Feature Importance (75 sec)

**Speaker:** Two complementary methods agree. Random Forest impurity importance puts
Elevation first at 0.243, followed by road distance, fire-point distance, and hydrology
distance. Permutation importance independently confirms this: shuffling Elevation drops
macro-F1 by 0.371 — the largest single-feature impact. The top four features are all
terrain-related, but wilderness areas and soil types also appear in the top 15, showing
the model uses contextual information beyond geometry alone.

### Slide 11 — Hyperparameter Tuning (60 sec)

**Speaker:** We ran grid search on 30k observations across three model classes.
All three degraded: DT 0.819→0.797, RF 0.908→0.886, HGB 0.898→0.889.
The natural question: is the 30k subset simply too small for any tuning?

To answer this, we ran Optuna (Bayesian TPE, 100 trials) on the DT
over the *same* 30k subset. Optuna found test F1=0.823 — recovering baseline
and beating grid search by 0.026. Same data, two search methods, very different outcomes.
The failure of grid search was NOT because the subset is too small, but because
its coarse 48-point grid missed the narrow region where CV and test performance align.
Bayesian optimization's continuous exploration and adaptive pruning found it.

Gain over defaults is marginal (+0.003), consistent with Probst et al.'s "flat
optimization surface" for tree ensembles. But the methodology lesson is clear:
grid search on a limited budget is fragile; Bayesian optimization is more robust
even on the same training data.

### Slide 12 — Calibration and Error Analysis (60 sec)

**Speaker:** Good classification does not guarantee reliable probabilities. We use
Brier score and Expected Calibration Error. Random Forest has the best per-class Brier
score at 0.024 (traditional multiclass Brier 0.171) but is systematically
under-confident: its mean top-label confidence is 0.783 while observed accuracy is 0.908.
HistGradientBoosting is best calibrated with ECE only 0.041. Platt (sigmoid) scaling
reduces RF Brier from 0.024 to 0.021. McNemar test ($\chi^2$=41.5, p<0.001) confirms
systematic error-pattern differences, but the macro-F1 gap is only 0.010 — bootstrap
CI shows it's stable but practically negligible. Performance also varies by elevation band: 0.863 accuracy in
the most diverse mid-elevation band versus 0.958 at the highest. It's not a simple
"higher is easier" story — the mixed mid-elevation zone is the hardest.

### Slide 13 — Advanced Model Comparison (60 sec)

**Speaker:** Looking beyond basic comparison: Random Forest variants consistently
outperform. Using top-30 features, RF achieves macro-F1 of 0.907. Standard sklearn
Gradient Boosting (120 iter, F1=0.883) is competitive in score but had significantly
longer training time — 160+ seconds vs under 2 seconds for RF. The histogram-based
variant (HGB, not shown here) achieves F1=0.898 with much faster training.
The trade-off favors Random Forest for this application:
strong performance with reasonable computational cost.

### Slide 14 — Conclusions and Limitations (45 sec)

**Speaker:** To summarize: Random Forest delivers 0.908 accuracy, 0.908 macro-F1,
and 0.992 macro-AUC. Bootstrap 95\% CIs confirm the F1 gap is stable but negligible
($\Delta$=0.010). McNemar ($\chi^2$=41.5, p<0.001) shows systematic per-sample error-pattern
differences between RF and HGB. Supervised learning is essential — 
clustering alone cannot recover the labels (best ARI≤0.215 with Gower distance, vs 
RF F1=0.908). Elevation is dominant, but RF on just 5 features reaches 0.877 F1 — 
nonlinearity, not dimensionality, drives performance. Polynomial expansion gave zero 
improvement for LR.

Caveats: modeling sample covers ~17% of full data (capped by minority class size).
Full-data gradient boosting in the literature reaches 0.95-0.96 accuracy — a gap
attributable to our sample-limited protocol.  Grid search on 30k subset degraded
all three model classes; however, Bayesian optimization (Optuna, 100 trials) on the
same subset recovered baseline for the Decision Tree — proving the bottleneck is the
search strategy, not the subset size.  Future work includes partial_fit for full-data
training and Bayesian search for the ensemble models.

### Slide 15 — Q&A (3 min reserved)

---

## Anticipated Q&A

**Q1: Why is clustering ARI so low (≤0.12) compared to RF (0.908)?**
A: Two structural reasons. First, the dataset mixes 10 continuous + 44 binary features:
Euclidean distance poorly captures similarity in mixed-type space, while tree-based
models use axis-aligned splits that handle binary indicators natively. We verified this
by testing Gower distance (designed for mixed data) with Ward linkage — ARI only
improved to 0.124, still an order of magnitude below RF. This proves the bottleneck
is the clustering objective itself, not the distance metric. Second, clustering
minimizes internal variance (silhouette), whereas ecological labels are defined by
forest management categories that don't align with variance-minimizing partitions.
The supervised model optimizes for label separation directly, which explains the huge gap.

**Q2: Why Random Forest over Gradient Boosting?**
A: RF achieves better macro-F1 (0.908 vs 0.898) and AUC (0.992 vs 0.991) on our
test set. Bootstrap 95% CIs do not overlap, confirming the F1 gap is stable, but
the absolute difference is only 0.010 — practically negligible. McNemar (χ²=41.5,
p<0.001) confirms the two models differ systematically in their per-sample correctness
patterns, but this tests accuracy-level agreement, not the F1 metric directly.
In practice, the models are interchangeable. HGB has better calibration (ECE 0.041
vs 0.125), while RF is faster (2 sec vs 160 sec). Choose based on deployment priority.

**Q3: How do you handle class imbalance?**
A: Stratified sampling preserves proportions. Class-weighted training adjusts loss.
Most importantly, we evaluate with macro-averaged metrics, not accuracy, so minority
classes are not ignored. SMOTE was explored but did not materially improve results
— class imbalance shares some signal with the terrain features.

**Q4: What's the computational cost of this analysis?**
A: Full-data t-SNE or Ward clustering would be prohibitive. We use stratified samples
for expensive steps, PCA before distance-based methods, MiniBatchKMeans for scalable
clustering, dtype downcasting (float32 + uint8, ~81% memory reduction), and the 44
binary features can additionally be stored in sparse CSR format. The entire notebook
runs within course-project resources. Incremental learning (partial_fit) would be
needed for full-dataset training.

**Q5: Which classes are hardest to predict and why?**
A: Lodgepole Pine has the lowest recall at 0.817, followed by Spruce/Fir at 0.842.
These dominant classes suffer most from the confusion between each other in
overlapping terrain zones. Aspen (0.965) and Krummholz (0.983) have the highest
recall, likely because they occupy distinct ecological niches (mid-elevation
deciduous stands and alpine tree-line, respectively).

**Q6: Why only ~100k modeling sample instead of full 581k?**
A: Stratified per-class sampling caps each class at the minority size (Cottonwood/Willow
has only 2,747 total). This ensures class balance but constrains the total sample to
~98k. For production deployment, partial_fit on the full dataset is recommended.

**Q7: Did Bayesian optimization help? How much?**
A: Yes. Grid search on 30k degraded DT from 0.819 to 0.797. Optuna (TPE, 100 trials)
on the same 30k subset recovered 0.823 — beating grid search by 0.026 and slightly
exceeding the baseline. The failure was not the subset size, but the coarse 48-point
grid missing the right region. Bayesian optimization's continuous exploration and
pruning found it. Gain over defaults is marginal (+0.003), consistent with the
"flat optimization surface" of tree ensembles (Probst et al., 2019).
