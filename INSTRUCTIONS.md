# Project Instructions — CS178 (Winter)

Follow these concrete tasks and deliverables to complete the project.

## Dataset

- Explore data: summary stats, class balance, and at least one visualization (e.g., sample images, histograms, confusion matrix example).

## Required Models (implement all)

- k‑Nearest Neighbors
- Logistic regression (multiclass if needed)
- Feedforward neural network with ≥1 hidden layer
- At least one additional classifier (e.g., Random Forest, Gradient Boosting, SVM, CNN for image, RNN for text)
- You may implement more models or ensembles if desired.

## Experimental Protocol

- Hold out a final test set (recommended 20%); use remaining 80% for training/validation.
  - Suggested split: training = 60% total, validation = 20% total (i.e., 75/25 on the 80%).
- Specify metrics: primary = accuracy; include others as appropriate (precision/recall/F1 for binary; per-class metrics for multiclass).
- Hyperparameter tuning: document ranges and search method (grid/random or cross‑validation). Use pseudocode or bullets to describe tuning steps.
- Reproducibility: record random seeds, software versions, environment, and exact data splits.

## Experiments & Reporting

- Report results on the held‑out test set only once at the end.
- Present results in a table or figure comparing models (test metric(s)), with brief interpretation.
- Optionally include learning curves, ablation studies, error analysis, and sample misclassifications.

## Final Report (PDF, ~4 pages)

Include numbered headings (in this order):

1. Title + Authors (with student IDs)
2. Summary (2–4 sentences)
3. Data Description (include ≥1 figure and cite any related paper)
4. Classifiers (short 1–2 sentence description per method; hyperparameter ranges; software used)
5. Experimental Setup (data splits, metrics, tuning procedure)
6. Experimental Results (table/figure and interpretation)
7. Insights / Error Analysis (what you learned)
8. Contributions (1–2 sentences per team member)

- Keep main report ≈4 pages; extra detail can go in an Appendix.
- Submit only the PDF to Canvas; include a link to your code repository in the report.

## Practical Recommendations

- Use scikit‑learn, PyTorch, TensorFlow, Pandas as appropriate.
- For CIFAR and convolutional nets, the PyTorch CIFAR tutorial is useful: <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>
- Do not post or share code on Ed; you may discuss approaches and links.

## Deliverables Checklist

- Team registered on Canvas
- GitHub repo with code (link in report)
- Final PDF report (~4 pages) with required sections and author contributions
- Reproducibility notes: seed, env, data split description

Complete each task early enough to allow experimentation and a clear writeup.
