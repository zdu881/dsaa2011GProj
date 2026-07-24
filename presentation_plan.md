# DSAA2011 Covertype Presentation Plan

## Time Allocation: 15 minutes total (12 min talk + 3 min Q&A)

## Key Metrics to Memorize

| Metric | Value |
|---|---|
| Dataset size | 581,012 rows × 54 features |
| Classes | 7 forest cover types (severe imbalance: 85.3% in top 2) |
| Best Model | Random Forest |
| Test Accuracy | 0.902 |
| Macro F1 | 0.845 |
| Macro OVR AUC | 0.991 |
| Clustering ARI vs labels | ~0 (clusters ≠ cover types) |
| Dominant feature | Elevation (RF impurity = 0.235) |

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

**Speaker:** We test MiniBatchKMeans and Agglomerative Ward clustering. Both select
K=2 as optimal based on silhouette score. MiniBatchKMeans achieves silhouette 0.191,
Ward achieves 0.153. The crucial number: adjusted Rand index against the true labels
is only 0.003–0.004 — effectively zero. This means the strongest unsupervised partition
does NOT correspond to the seven ecological labels. Clustering reveals broad terrain
structure, not cover type boundaries. So for prediction, we must use supervised methods.

### Slide 7 — Simple Prediction Models (90 sec)

**Speaker:** We first train two baseline classifiers. Logistic regression — a linear
baseline — gets test accuracy 0.593 and macro-F1 only 0.471. It reaches macro AUC 0.923,
so the probability ranking is useful but class decisions are weak. The decision tree,
with nonlinear splits, raises accuracy to 0.775 and macro-F1 to 0.663. However, the
train-test gap shows moderate overfitting. Both are evaluated on the full 581K dataset
after training, with consistent results. These baselines tell us: the problem requires
ensembles that reduce variance while preserving nonlinear capability.

### Slide 8 — Model Evaluation and Choice (90 sec)

**Speaker:** We extend to Random Forest and HistGradientBoosting. Random Forest wins
across the board: accuracy 0.902, macro-F1 0.845, macro AUC 0.991. HistGradientBoosting
follows at 0.831 macro-F1 with the highest macro precision — fewer false positive
minority predictions. Cross-validation on a separate 36,000-row sample confirms the
same ranking, so the result is not split-dependent. The high AUC across all models
confirms that cartographic features contain strong signal. We select Random Forest
as the best model, balancing majority and minority class performance.

### Slide 9 — Open-Ended Exploration Overview (45 sec)

**Speaker:** Beyond baseline models, we pursue six exploration directions: feature
importance, hyperparameter tuning, feature group ablation, calibration, error analysis,
and model interpretability. These collectively validate that topographic factors —
especially elevation — drive predictions, while soil and wilderness context provide
complementary signal. We also verify that complex tuning is not always beneficial.

### Slide 10 — Feature Importance (75 sec)

**Speaker:** Two complementary methods agree. Random Forest impurity importance puts
Elevation first at 0.235, followed by road distance, fire-point distance, and hydrology
distance. Permutation importance independently confirms this: shuffling Elevation drops
macro-F1 by 0.396 — the largest single-feature impact. The top four features are all
terrain-related, but wilderness areas and soil types also appear in the top 15, showing
the model uses contextual information beyond geometry alone.

### Slide 11 — Hyperparameter Tuning (60 sec)

**Speaker:** We ran a grid search over 48 decision tree configurations with 3-fold
cross-validation. The best config achieved CV-F1 of 0.769. However — applying those
tuned parameters to the full test set DECREASED performance. Why? The tuning was on
a limited subset, and the optimal subset parameters didn't generalize. The original
parameters — depth 24, leaf 15 — were already near-optimal. This is an important
practical lesson: hyperparameter tuning is not universally beneficial, especially
when the baseline is well-chosen. For the Random Forest, increasing trees beyond 120
yielded only marginal gains.

### Slide 12 — Calibration and Error Analysis (60 sec)

**Speaker:** Good classification does not guarantee reliable probabilities. We use
Brier score and Expected Calibration Error. Random Forest has the best Brier score at
0.178 but is systematically under-confident: its mean confidence is 0.775 while
observed accuracy is 0.902. HistGradientBoosting is best calibrated with ECE only 0.047.
Importantly, incorrect predictions have substantially lower confidence — mean 0.595 vs
0.794 for correct ones. So low confidence can flag uncertain predictions for review.
Performance also improves with elevation: 0.874 accuracy in the lowest band versus
0.938 in the highest.

### Slide 13 — Advanced Model Comparison (60 sec)

**Speaker:** Looking beyond basic comparison: Random Forest variants consistently
outperform. Using top-30 features, RF achieves macro-F1 of 0.845. Gradient Boosting
is competitive in score but had significantly longer training time — 160+ seconds vs
under 30 seconds for RF. The trade-off favors Random Forest for this application:
strong performance with reasonable computational cost.

### Slide 14 — Conclusions and Future Work (45 sec)

**Speaker:** To summarize: Random Forest delivers 0.902 accuracy, 0.845 macro-F1,
and 0.991 macro-AUC. Supervised learning is essential — clustering alone cannot
recover the labels. Elevation is the dominant predictor, but distances to roads,
fire points, water, plus soil and wilderness context collectively improve predictions.
Future work includes deeper feature engineering of terrain-climate interactions,
multi-model fusion with stacking, and targeted optimization for minority classes
like Aspen.

### Slide 15 — Q&A (3 min reserved)

---

## Anticipated Q&A

**Q1: Why is clustering ARI nearly zero?**
A: The internal structure favors two coarse groups based on terrain (essentially
high vs. low elevation), while the labels differentiate seven ecologically distinct
types. Unsupervised methods find the strongest variance partition; supervised methods
learn the labeled concept. They answer different questions.

**Q2: Why Random Forest over Gradient Boosting?**
A: RF achieves better macro-F1 (0.845 vs 0.831) and AUC (0.991 vs 0.985) on our
test set. Boosting has better calibration but lower minority-class recall. For
classification ranking, RF wins. For probability interpretation, Boosting is better.

**Q3: How do you handle class imbalance?**
A: Stratified sampling preserves proportions. Class-weighted training adjusts loss.
Most importantly, we evaluate with macro-averaged metrics, not accuracy, so minority
classes are not ignored. SMOTE was explored but did not materially improve results
— class imbalance shares some signal with the terrain features.

**Q4: What's the computational cost of this analysis?**
A: Full-data t-SNE or Ward clustering would be prohibitive. We use stratified samples
for expensive steps, PCA before distance-based methods, MiniBatchKMeans for scalable
clustering, dtype downcasting (81% memory reduction), and cache manifests for
reproducibility. The entire notebook runs within course-project resources.

**Q5: Which classes are hardest to predict and why?**
A: Aspen has the lowest recall at 0.639, followed by Douglas-fir at 0.778. Both are
ecologically adjacent to dominant classes — Aspen overlaps with Lodgepole Pine in
mid-elevation zones, and Douglas-fir can be confused with Ponderosa Pine. Lower
elevation bands are also harder because the class mixture is more diverse there.
