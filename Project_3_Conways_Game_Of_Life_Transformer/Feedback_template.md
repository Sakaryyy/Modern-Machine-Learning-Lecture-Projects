# ML Mini-Project 3 Feedback

**Student(s):**

**Course:** Modern Machine Learning – Project 3

**Date:** 09.01.2026

---

## Points Summary

| Block                                      |    Max | Points    |
|--------------------------------------------|-------:|:----------|
| Task 1: Sequence Learning & Generalization |      8 | __/8      |
| Task 2: Stochastic Dynamics & Setup        |      6 | __/6      |
| Task 3: Anomaly Detection Results          |      6 | __/6      |
| Presentation                               |      5 | __/5      |
| **Total**                                  | **25** | **__/25** |

---

## Task 1: Sequence Learning & Generalization (8 pts)

**Requirement:** Train a sequence-to-sequence model to learn the deterministic Game of Life rules. The model must be
trained on the initial grid size and evaluated for generalization on larger grid sizes.

### 1) Model Architecture and Training Setup (4)

- [ ] Model architecture is clearly described and justified (for example CNN, ResNet, Encoder-Decoder or other
  sequence-to-sequence approaches suitable for 2D grids).
- [ ] Training process is detailed (for example loss function, optimization etc.).
- [ ] Training convergence is demonstrated (for example via learning curves) on the base grid size (
  e.g., $16 \times 16$).
- [ ] Data generation for the deterministic rule pairs $(x, x')$ is explained and quantified.

**Points (0–4):** __/4

**Feedback:**

### 2) Generalization Evaluation (4)

- [ ] The model is evaluated on larger grid sizes (e.g., $32 \times 32$) to test generalization capabilities.
- [ ] Quantitative performance metrics (accuracy, loss) or qualitative visualizations are provided for the larger grids.
- [ ] There is a discussion on whether the model generalizes to unseen grid dimensions and why (like padding for CNN
  used or boundary conditions etc.).

**Points (0–4):** __/4

**Feedback:**

**Subtotal – Task 1:** __/8

---

## Task 2: Stochastic Dynamics & Setup (6 pts)

**Requirement:** Adaptation of the model to learn stochastic update rules and setup of the anomaly detection methodology
using likelihood estimates.

### 1) Stochastic Training (3)

- [ ] Implementation of the stochastic transition rules (flipping between rule sets based on probability $p$) is
  correct.
- [ ] A model is trained or adapted to predict updates under the specific stochastic dynamics defined in the
  task ($p=0.8$).
- [ ] Training results or validation metrics for the stochastic model are reported.

**Points (0–3):** __/3

**Feedback:**

### 2) Anomaly Detection Methodology (3)

- [ ] The method for computing the likelihood $P_\theta(x'|x)$ or a comparable anomaly score is included.
- [ ] The test dataset composition (ratio of normal dynamics to anomalous dynamics) corresponds to the project
  specifications.
- [ ] The thresholding approach for classifying anomalies based on the calculated scores is defined.

**Points (0–3):** __/3

**Feedback:**

**Subtotal – Task 2:** __/6

---

## Task 3: Anomaly Detection Results (6 pts)

**Requirement:** Quantitative evaluation of anomaly detection performance using ROC curves across varying grid sizes and
analysis of the results.

### 1) ROC Curves and Metrics (3)

- [ ] True Positive Rate (TPR) and False Positive Rate (FPR) are calculated correctly.
- [ ] ROC curves are plotted to visualize the tradeoff between TPR and FPR.
- [ ] Evaluation is performed across the multiple grid sizes requested (like small, medium, and large grids).

**Points (0–3):** __/3

**Feedback:**

### 2) Analysis and Discussion (3)

- [ ] The observed behavior of the ROC curves is discussed (like area under the curve, shape).
- [ ] The impact of increasing grid size on anomaly detection performance is analyzed.
- [ ] Plausible explanations for the observations are provided (for example how likelihood scaling affects separability
  of anomalies).

**Points (0–3):** __/3

**Feedback:**

**Subtotal – Task 3:** __/6

---

## Presentation (5 pts)

**Requirement:** Clear, concise report with expressive figures.

- [ ] Figures are clear
- [ ] Tables are clear
- [ ] Captions are informative and self-contained
- [ ] Layout is coherent and easy to follow
- [ ] Language is precise and concise

**Points (0–5):** __/5

**Feedback:**

---

## General Feedback

* 