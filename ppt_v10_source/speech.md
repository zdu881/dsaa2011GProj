# Forest Cover Type Analysis — 8-minute presentation script

Designed for a non-native speaker: about 650 spoken words, with time for pointing, changing slides, and short pauses. Text in square brackets is an action cue and should not be spoken.

## Slide 1 — Title | 0:00–0:20

Good morning. We are Hao Bai, Siqi Lin, Yijie Wang, and Yuling Zhang. Today, we will present our analysis of forest cover types. We will introduce the data, compare several models, and explain our main findings.

[Pause. Change slide.]

## Slide 2 — Dataset | 0:20–0:55

This dataset contains more than 581 thousand forest observations, 54 features, and seven cover types.

[Point to the class-distribution chart.]

The main challenge is class imbalance. Classes 1 and 2 represent 85.3 percent of the data. Therefore, we focus on macro F1, because it gives equal importance to every class. We use a representative sample for most experiments.

## Slide 3 — Cover types | 0:55–1:15

These are the seven forest cover types.

[Move your hand from left to right across the cards.]

They represent different ecological communities. However, their terrain conditions can overlap, so some classes are naturally difficult to separate.

## Slide 4 — Preprocessing | 1:15–1:40

The dataset has no missing values. We standardize the continuous features, but keep the soil and wilderness indicators as binary values.

[Point to the before-and-after plots.]

To prevent data leakage, scaling is fitted only on the training data.

## Slide 5 — t-SNE | 1:40–2:10

We first explore the data with PCA and t-SNE.

[Point to two or three colored regions.]

The plot shows some local structure, but the colors overlap. This means the seven labels do not form seven clearly separated geometric groups. We use this plot only as exploratory evidence, not as proof of a cluster structure.

## Slide 6 — Clustering | 2:10–2:50

The clustering results confirm this problem.

[Point first to the left chart, then to the dendrogram.]

K-means prefers three clusters, while hierarchical clustering prefers four. But their adjusted Rand scores are only 0.075 and 0.103. Gower distance improves the result to 0.215, but it is still low. The data contain natural terrain patterns, but these patterns do not match the seven forest labels very well.

## Slide 7 — Baseline models | 2:50–3:25

Supervised learning performs much better.

[Point to Logistic Regression, then Decision Tree.]

Logistic Regression achieves a macro F1 of 0.660. The Decision Tree improves this to 0.819. This large difference shows that the relationship between terrain and forest type is nonlinear and depends on feature interactions.

## Slide 8 — Model selection | 3:25–4:20

Here is our main model comparison.

[Point to the Random Forest row.]

Random Forest performs best, with 0.908 accuracy, 0.908 macro F1, and 0.992 AUC. Histogram Gradient Boosting is very close, with macro F1 of 0.898.

[Point to the two charts on the right. Pause briefly.]

The validation and confidence-interval results support the same ranking. The difference is statistically detectable, but the practical gap is only about one percentage point. We therefore select Random Forest, while keeping gradient boosting as a strong alternative.

## Slide 9 — Further analysis | 4:20–4:30

Next, we look beyond the overall score and examine importance, tuning, calibration, and subgroup performance.

## Slide 10 — Feature importance | 4:30–5:05

[Point to the longest bars.]

Elevation is the most important feature in both ensemble models. Distance to roads and fire points also provides useful information. The top-five-feature Random Forest still reaches 0.877 macro F1, while polynomial Logistic Regression reaches only 0.584. So model choice matters more than simply adding polynomial features.

## Slide 11 — Hyperparameter search | 5:05–5:40

[Point to the heatmap and then the three boxes.]

A larger grid search does not automatically produce a better model. In our experiment, grid search slightly reduces performance. Optuna recovers a Decision Tree macro F1 of 0.823 on the same tuning subset. The lesson is that a smarter search strategy is more useful than simply testing more combinations.

## Slide 12 — Calibration | 5:40–6:20

Accuracy is not the only measure of model quality.

[Trace the diagonal line on the calibration chart.]

Gradient Boosting has the best calibration. Random Forest is under-confident, although additional calibration improves its Brier score. Confidence is still useful: correct predictions have average confidence of 0.811, compared with 0.586 for incorrect predictions.

[Point to the lower-left distribution and pause.]

Low-confidence cases could therefore be flagged for review.

## Slide 13 — Robustness | 6:20–7:05

We also check performance across different groups.

[Point to the three charts from left to right.]

Results change across elevation bands, forest classes, and wilderness areas. Lodgepole Pine and Spruce/Fir have relatively low recall. Cache la Poudre is the most difficult wilderness area. These results show that one overall score can hide important local weaknesses.

## Slide 14 — Conclusion | 7:05–7:50

To conclude, unsupervised clustering does not recover the seven forest labels well, but nonlinear supervised models perform strongly.

[Point to the key-conclusions box.]

Random Forest gives the best overall prediction, while Gradient Boosting provides similar performance and better calibration. Elevation is the strongest feature, and the remaining errors are concentrated in particular classes and regions.

[Point to the future-work box.]

Future work should use the full dataset, spatial validation, improved calibration, and more attention to minority classes.

## Slide 15 — Q&A | 7:50–8:00

Thank you for listening. We are happy to take your questions.
