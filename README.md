# Adaptive Continual Learning for Non-Stationary Data

# 1. Abstract

Modern deep-learning systems are commonly trained using static datasets under the assumption that the underlying data distribution remains relatively stable. However, real-world applications continuously receive new data, new classes, new concepts, or changing distributions.

When a conventional neural network is sequentially fine-tuned on newly arriving data, it may adapt to the new information while losing previously learned knowledge. This phenomenon is known as **catastrophic forgetting**.

This project proposes an investigation into **adaptive continual learning for non-stationary data**, focusing on the trade-off between learning new knowledge and retaining previously acquired knowledge.

The project will first establish strong baselines using conventional fine-tuning, replay-based methods, regularization, knowledge distillation, and parameter-efficient learning. Their performance will be evaluated using accuracy, forgetting, forward/backward transfer, memory consumption, training cost, and parameter efficiency.

Based on the literature review and baseline experiments, an **adaptive knowledge-retention mechanism** will be investigated. The mechanism will dynamically determine how aggressively previously acquired knowledge should be preserved based on factors such as distribution shift, task similarity, available memory, and model performance.

The final objective is to develop a reproducible continual-learning framework and investigate whether adaptive knowledge retention can provide a better balance between **stability, plasticity, memory efficiency, and computational cost**.

---

# 2. Problem Statement

Traditional deep-learning training follows approximately:

```text
Static Dataset
      ↓
Model Training
      ↓
Trained Model
      ↓
Deployment
```

Real-world systems instead encounter:

```text
Data₁ → Data₂ → Data₃ → Data₄ → Data₅ → ...
```

where the distribution of the incoming data can change over time.

A naïve approach is:

```text
Train on Data₁
      ↓
Fine-tune on Data₂
      ↓
Fine-tune on Data₃
      ↓
Fine-tune on Data₄
```

This can cause:

```text
New knowledge increases
        ↓
Old knowledge decreases
        ↓
Catastrophic Forgetting
```

Therefore, the central problem is:

> **How can a deep-learning model continuously learn new information while retaining previously acquired knowledge without requiring complete retraining and while operating under realistic memory and computational constraints?**

---

# 3. Research Question

## Primary Research Question

> Can an adaptive knowledge-retention mechanism reduce catastrophic forgetting while maintaining effective learning of new information under constrained memory and computational resources?

## Secondary Questions

### RQ1

How do existing continual-learning strategies differ in their ability to balance stability and plasticity?

### RQ2

How does the severity of distribution shift affect catastrophic forgetting?

### RQ3

How does replay-memory size affect the trade-off between forgetting and computational/memory cost?

### RQ4

Can task/distribution similarity be used to dynamically determine the amount of knowledge that should be retained?

### RQ5

Can adaptive retention achieve competitive performance with lower memory or computational requirements than fixed continual-learning strategies?

---

# 4. Research Hypothesis

### H1

An adaptive knowledge-retention strategy can reduce catastrophic forgetting compared with naïve sequential fine-tuning.

### H2

Adaptive retention can provide a better stability-plasticity trade-off than using a fixed retention strategy across all tasks.

### H3

The proposed approach can maintain competitive accuracy while reducing the memory or computational requirements associated with continual learning.

---

# 5. Research Objectives

## O1 — Literature and Research Gap Analysis

Study existing approaches to:

* Continual Learning
* Class-Incremental Learning
* Task-Incremental Learning
* Domain-Incremental Learning
* Distribution Shift
* Catastrophic Forgetting
* Experience Replay
* Regularization
* Knowledge Distillation
* Parameter-Efficient Learning
* Selective Forgetting
* Open-World Learning

### Output

```text
Literature Matrix
       ↓
Existing Methods
       ↓
Limitations
       ↓
Research Gap
```

---

## O2 — Develop a Reproducible Baseline Framework

Implement common continual-learning baselines under a unified experimental environment.

Baselines:

```text
1. Naïve Fine-Tuning
2. Experience Replay
3. Regularization
4. Knowledge Distillation
5. Parameter-Efficient Adaptation
```

---

## O3 — Quantify Catastrophic Forgetting

Measure:

* Average Accuracy
* Forgetting
* Forward Transfer
* Backward Transfer
* Task-wise Accuracy
* Final Accuracy

---

## O4 — Analyze Stability vs Plasticity

Study the trade-off between:

```text
STABILITY
Preserve old knowledge
        ↕
PLASTICITY
Learn new knowledge
```

---

## O5 — Investigate Adaptive Knowledge Retention

Develop a mechanism that dynamically determines how much previous knowledge should be retained based on:

```text
Task Similarity
       +
Distribution Shift
       +
Current Performance
       +
Memory Budget
       +
Computational Budget
```

---

## O6 — Evaluate Resource Efficiency

Measure:

* GPU memory
* CPU/RAM usage
* Training time
* Inference time
* Number of trainable parameters
* Replay-memory size
* Model size

---

## O7 — Conduct Ablation Studies

Determine which components actually contribute to performance.

```text
Full Model
   ↓
Remove Component A
   ↓
Remove Component B
   ↓
Remove Component C
   ↓
Compare
```

---

## O8 — Produce a Research Publication

The final research output should contain:

```text
Problem
   ↓
Literature Review
   ↓
Research Gap
   ↓
Method
   ↓
Experiments
   ↓
Ablation
   ↓
Analysis
   ↓
Conclusion
```

The goal is **publication potential**, not a guaranteed publication.

---

# 6. Proposed System Architecture

```text
                         ┌─────────────────────┐
                         │   Data Stream       │
                         │ D1 → D2 → D3 → D4  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Data Preprocessing  │
                         │ Normalization       │
                         │ Validation          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │ Distribution Shift Analyzer  │
                    │                              │
                    │ Task Similarity              │
                    │ Feature Distribution         │
                    │ Performance Change            │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Adaptive Retention Controller│
                    │                              │
                    │ Retention Strength           │
                    │ Replay Size                  │
                    │ Distillation Weight          │
                    │ Regularization Weight        │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
       ┌─────────────┐      ┌──────────────┐    ┌──────────────┐
       │ Replay      │      │ Regularizer  │    │ Distillation │
       │ Memory      │      │              │    │              │
       └──────┬──────┘      └──────┬───────┘    └──────┬───────┘
              │                    │                   │
              └────────────────────┼───────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ Continual Learner   │
                         │                     │
                         │ Backbone + Adapter  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Updated Model       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Evaluation Engine   │
                         ├─────────────────────┤
                         │ Accuracy            │
                         │ Forgetting          │
                         │ Transfer            │
                         │ Memory              │
                         │ Training Cost       │
                         └─────────────────────┘
```

---

# 7. Core Research Concept

The project revolves around:

```text
             NEW KNOWLEDGE
                   ▲
                   │
                   │
        Stability ↔ Plasticity
                   │
                   │
                   ▼
             OLD KNOWLEDGE
```

The model must avoid two extreme cases.

### Too much plasticity

```text
New task learned very well
        ↓
Old knowledge destroyed
```

### Too much stability

```text
Old knowledge preserved
        ↓
New task learned poorly
```

The desired point is:

```text
          Optimal Balance
               ↓
    ┌──────────┴──────────┐
    ↓                     ↓
Retain old            Learn new
knowledge             information
```

---

# 8. Baseline Methods

## Baseline 1 — Naïve Fine-Tuning

```text
Model₁
  ↓
Train on Task₂
  ↓
Model₂
  ↓
Train on Task₃
  ↓
Model₃
```

Purpose:

> Establish the catastrophic-forgetting baseline.

---

# 9. Baseline 2 — Experience Replay

Maintain a limited memory:

```text
Memory Buffer
 ├── Old Sample 1
 ├── Old Sample 2
 ├── Old Sample 3
 └── ...
```

During new-task training:

```text
New Data + Replay Data
          ↓
       Training
```

Research variable:

```text
Memory = 0%
Memory = 1%
Memory = 5%
Memory = 10%
```

This allows analysis of:

> How much historical data is actually necessary?

---

# 10. Baseline 3 — Regularization

Add a penalty discouraging harmful changes to important parameters.

Conceptually:

```text
Total Loss
     =
New Task Loss
     +
Knowledge Retention Loss
```

Investigate the effect of different retention strengths.

---

# 11. Baseline 4 — Knowledge Distillation

Previous model:

```text
Teacher
   ↓
Old knowledge
```

Current model:

```text
Student
   ↓
New knowledge
```

The student learns from:

```text
Ground Truth
      +
Teacher Knowledge
```

---

# 12. Baseline 5 — Parameter-Efficient Learning

Instead of updating the entire model:

```text
Full Model
██████████████████
```

use:

```text
Frozen Backbone
██████████████████
        +
Trainable Adapter
     ██
```

Measure whether parameter-efficient adaptation reduces computational cost while maintaining retention.

---

# 13. Proposed Adaptive Component

This is the part that requires literature review before being finalized.

The conceptual design is:

```text
Incoming Task
      ↓
Distribution Analysis
      ↓
Task Similarity
      ↓
Current Model Performance
      ↓
Available Memory
      ↓
Adaptive Controller
      ↓
┌─────┴───────────────────┐
│                         │
▼                         ▼
High similarity       Low similarity
│                         │
Lower retention       Higher retention
cost                   requirement
│                         │
└──────────┬──────────────┘
           ↓
     Model Update
```

The controller could dynamically determine:

```text
Replay Size
Distillation Weight
Regularization Strength
Adapter Size
Learning Rate
```

The exact mechanism should be selected **after the literature review**, so the project does not make an unsupported novelty claim.

---

# 14. Experimental Setup

## Phase 1 — Controlled Benchmark

Start with:

```text
CIFAR-10
CIFAR-100
```

Potential extension:

```text
TinyImageNet
```

The initial experiments should use controlled class-incremental or domain-incremental settings.

---

# 15. Sequential Learning Setup

Example:

```text
Task 1
Classes 0–1
       ↓
Task 2
Classes 2–3
       ↓
Task 3
Classes 4–5
       ↓
Task 4
Classes 6–7
       ↓
Task 5
Classes 8–9
```

At every stage:

```text
Train
 ↓
Evaluate ALL previous tasks
 ↓
Record results
 ↓
Continue
```

This is critical.

Do not only evaluate the newest task.

---

# 16. Evaluation Metrics

## Accuracy

Measure performance on:

```text
Current Task
+
Previous Tasks
```

---

## Forgetting

For each previous task:

```text
Maximum Previous Accuracy
          -
Current Accuracy
```

Lower forgetting is desirable.

---

## Average Accuracy

Average performance across all learned tasks.

---

## Forward Transfer

Measure whether previously learned knowledge helps learning future tasks.

---

## Backward Transfer

Measure whether learning new tasks affects previously learned tasks.

---

## Memory Cost

```text
Replay samples
+
Model parameters
+
GPU memory
```

---

## Computational Cost

Measure:

```text
Training time
Inference time
FLOPs where practical
Number of trainable parameters
```

---

# 17. Main Experimental Matrix

```text
                     Fine   Replay   Reg.   Distill   PEFT   Proposed
----------------------------------------------------------------------
Accuracy
Forgetting
Forward Transfer
Backward Transfer
Memory
Training Time
Inference Time
Parameters
```

The proposed method should not simply beat every baseline on every metric.

The research question is whether it provides a **better overall trade-off**.

---

# 18. Ablation Study

A full proposed system may contain:

```text
A = Task Similarity
B = Distribution Shift
C = Replay
D = Distillation
E = Adaptive Controller
```

Run:

```text
Full Model
Full - A
Full - B
Full - C
Full - D
Full - E
```

Then determine:

> Which components actually matter?

This is important for a research paper.

---

# 19. Stress Testing

After controlled experiments, introduce harder conditions.

### Scenario 1

Small memory.

```text
1%
```

### Scenario 2

Large distribution shift.

```text
Task A → Very different Task B
```

### Scenario 3

Similar tasks.

```text
Task A → Similar Task B
```

### Scenario 4

Long task sequence.

```text
T1 → T2 → T3 → ... → T10
```

### Scenario 5

Class imbalance.

### Scenario 6

Noisy data.

This determines whether the method is robust rather than simply optimized for one benchmark.

---

# 20. Tech Stack

## Core

```text
Python 3.11+
PyTorch
Torchvision
NumPy
Pandas
Scikit-learn
```

## Research

```text
Hugging Face
timm
SciPy
Matplotlib
Seaborn
```

## Experiment Management

```text
MLflow
Weights & Biases
TensorBoard
Hydra
YAML
```

## Development

```text
Git
GitHub
VS Code
Jupyter
Docker
```

## Hardware

```text
CUDA-enabled GPU
```

The framework should also support CPU execution for reproducibility.

---

# 21. Software Architecture

```text
                    ┌──────────────────┐
                    │ Configuration    │
                    │ YAML / Hydra     │
                    └────────┬─────────┘
                             ↓
┌─────────────────────────────────────────────────────┐
│                  EXPERIMENT ENGINE                  │
│                                                     │
│ Dataset → Stream → Learner → Evaluator → Logger    │
└─────────────────────────────────────────────────────┘
         │             │            │
         ↓             ↓            ↓
    Dataset       Continual      Metrics
    Manager        Strategy       Engine
                       │
          ┌────────────┼─────────────┐
          ↓            ↓             ↓
       Replay      Regularizer   Distillation
          │            │             │
          └────────────┼─────────────┘
                       ↓
                Adaptive Controller
                       ↓
                   Model
                       ↓
                  Checkpoint
```

---

# 22. GitHub Repository Architecture

```text
adaptive-continual-learning/
│
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── pyproject.toml
│
├── configs/
│   ├── baseline/
│   │   ├── finetune.yaml
│   │   ├── replay.yaml
│   │   ├── regularization.yaml
│   │   ├── distillation.yaml
│   │   └── peft.yaml
│   │
│   ├── experiments/
│   │   ├── cifar10.yaml
│   │   ├── cifar100.yaml
│   │   └── tinyimagenet.yaml
│   │
│   └── proposed/
│       └── adaptive_retention.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── task_stream.py
│   │   └── distribution_shift.py
│   │
│   ├── models/
│   │   ├── backbone.py
│   │   ├── classifier.py
│   │   └── adapters.py
│   │
│   ├── continual/
│   │   ├── base.py
│   │   ├── finetune.py
│   │   ├── replay.py
│   │   ├── regularization.py
│   │   ├── distillation.py
│   │   ├── peft.py
│   │   └── adaptive.py
│   │
│   ├── memory/
│   │   ├── buffer.py
│   │   ├── sampler.py
│   │   └── policies.py
│   │
│   ├── metrics/
│   │   ├── accuracy.py
│   │   ├── forgetting.py
│   │   ├── transfer.py
│   │   └── efficiency.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   └── scheduler.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── benchmark.py
│   │   └── ablation.py
│   │
│   └── utils/
│       ├── logging.py
│       ├── reproducibility.py
│       └── checkpoint.py
│
├── experiments/
│   ├── baseline/
│   ├── proposed/
│   └── ablation/
│
├── results/
│   ├── raw/
│   ├── processed/
│   ├── tables/
│   └── figures/
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_baseline_analysis.ipynb
│   ├── 03_forgetting_analysis.ipynb
│   ├── 04_ablation_analysis.ipynb
│   └── 05_final_results.ipynb
│
├── docs/
│   ├── proposal.md
│   ├── literature_review.md
│   ├── methodology.md
│   ├── experiments.md
│   └── publication.md
│
└── papers/
    ├── figures/
    ├── tables/
    └── manuscript/
```

---

# 23. Research Workflow

```text
                    START
                      │
                      ▼
              Literature Review
                      │
                      ▼
             Identify Research Gap
                      │
                      ▼
            Define Research Question
                      │
                      ▼
             Build Baseline System
                      │
                      ▼
             Run Baseline Experiments
                      │
                      ▼
             Analyze Weaknesses
                      │
                      ▼
             Design Proposed Method
                      │
                      ▼
               Implement Method
                      │
                      ▼
             Compare With Baselines
                      │
                      ▼
               Ablation Studies
                      │
                      ▼
              Robustness Testing
                      │
                      ▼
            Statistical Evaluation
                      │
                      ▼
             Research Conclusions
                      │
                      ▼
                Paper Draft
                      │
                      ▼
                 Submission
```

---

# 24. Complete Roadmap

## Phase 0 — Professor Discussion

**Duration:** 1 week

Deliverables:

```text
1-page proposal
Research question
Initial architecture
Initial bibliography
```

Goal:

> Get supervisor approval before investing heavily in implementation.

---

# Phase 1 — Literature Review

**Duration:** 2–3 weeks

Study approximately:

```text
Continual Learning
Catastrophic Forgetting
Experience Replay
EWC / Regularization
Knowledge Distillation
Class-Incremental Learning
Domain-Incremental Learning
Parameter-Efficient Learning
Selective Forgetting
Open-World Learning
Distribution Shift
```

Create a table:

| Paper | Problem | Method | Dataset | Metric | Limitation | Possible Gap |
| ----- | ------- | ------ | ------- | ------ | ---------- | ------------ |

Deliverable:

```text
docs/literature_review.md
```

---

# Phase 2 — Research Gap Definition

**Duration:** 1 week

Do not decide the final novelty before this stage.

Output:

```text
Existing Method
       ↓
Limitation
       ↓
Research Gap
       ↓
Research Question
       ↓
Proposed Solution
```

Deliverable:

```text
docs/research_gap.md
```

---

# Phase 3 — Framework Development

**Duration:** 2–3 weeks

Build:

```text
Dataset Manager
Task Stream
Training Engine
Evaluation Engine
Experiment Logger
```

First goal:

> Run one complete continual-learning experiment automatically.

---

# Phase 4 — Baselines

**Duration:** 3–4 weeks

Implement:

```text
Fine-Tuning
Replay
Regularization
Distillation
PEFT
```

Generate:

```text
Accuracy curves
Forgetting curves
Memory comparison
Training-time comparison
```

---

# Phase 5 — Research Analysis

**Duration:** 2 weeks

Identify:

```text
Where does each method fail?
Why does it fail?
Under what conditions?
```

This stage determines whether the proposed adaptive method needs to change.

---

# Phase 6 — Proposed Method

**Duration:** 4–6 weeks

Develop:

```text
Adaptive Controller
       ↓
Retention Decision
       ↓
Continual Learner
```

Run controlled experiments.

---

# Phase 7 — Ablation

**Duration:** 2–3 weeks

Test:

```text
Full Model
      vs
Without Similarity
      vs
Without Replay
      vs
Without Distillation
      vs
Without Adaptive Controller
```

---

# Phase 8 — Robustness

**Duration:** 2–3 weeks

Test:

```text
Different datasets
Different task sequences
Different memory budgets
Different distribution shifts
Different random seeds
```

---

# Phase 9 — Final Evaluation

**Duration:** 1–2 weeks

Produce:

```text
Final Tables
Final Graphs
Statistical Analysis
Error Analysis
Computational Analysis
```

---

# Phase 10 — Paper

**Duration:** 3–4 weeks

Paper structure:

```text
1. Abstract
2. Introduction
3. Related Work
4. Problem Formulation
5. Proposed Method
6. Experimental Setup
7. Results
8. Ablation Study
9. Discussion
10. Limitations
11. Conclusion
```

---

# 25. Expected Figures for the Paper

The repository should eventually generate:

### Figure 1

System architecture.

### Figure 2

Continual-learning pipeline.

### Figure 3

Accuracy over sequential tasks.

### Figure 4

Catastrophic forgetting comparison.

### Figure 5

Memory vs performance.

### Figure 6

Training cost vs performance.

### Figure 7

Ablation study.

### Figure 8

Effect of distribution shift.

### Figure 9

Effect of replay-memory size.

### Figure 10

Stability-plasticity trade-off.

---

# 26. Expected Results Table

The final paper should contain something like:

| Method         | Avg. Accuracy | Forgetting ↓ | FWT ↑ | BWT ↑ | Memory ↓ | Time ↓ |
| -------------- | ------------: | -----------: | ----: | ----: | -------: | -----: |
| Fine-tuning    |               |              |       |       |          |        |
| Replay         |               |              |       |       |          |        |
| Regularization |               |              |       |       |          |        |
| Distillation   |               |              |       |       |          |        |
| PEFT           |               |              |       |       |          |        |
| Proposed       |               |              |       |       |          |        |

The numbers must come from experiments. Never manufacture expected results.

---

# 27. Reproducibility Requirements

Every experiment should record:

```text
Random seed
Dataset version
Model architecture
Hyperparameters
Learning rate
Batch size
Number of epochs
Task sequence
Memory budget
GPU
Python version
PyTorch version
Git commit
```

A result should be reproducible using:

```bash
python train.py \
    --config configs/proposed/adaptive_retention.yaml \
    --seed 42
```

---

# 28. GitHub README Structure

The README should eventually contain:

```text
# Adaptive Continual Learning

## Overview

## Research Problem

## Research Question

## Proposed Approach

## Architecture

## Installation

## Dataset

## Training

## Evaluation

## Results

## Reproducibility

## Research Timeline

## Publication

## Citation

## Contributors

## License
```

---

# 29. Initial Commands

Example repository workflow:

```bash
git clone <repository>
cd adaptive-continual-learning

python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

Run baseline:

```bash
python -m src.training.trainer \
    --config configs/baseline/finetune.yaml
```

Run replay:

```bash
python -m src.training.trainer \
    --config configs/baseline/replay.yaml
```

Run proposed method:

```bash
python -m src.training.trainer \
    --config configs/proposed/adaptive_retention.yaml
```

Run evaluation:

```bash
python -m src.evaluation.benchmark \
    --results results/raw/
```

---

# 30. Research Deliverables

At the end of the project:

### Software

```text
✓ Reproducible continual-learning framework
✓ Baseline implementations
✓ Proposed method
✓ Evaluation engine
✓ Visualization pipeline
✓ Configuration system
```

### Research

```text
✓ Literature review
✓ Research gap
✓ Problem formulation
✓ Experimental protocol
✓ Baseline comparison
✓ Proposed method
✓ Ablation study
✓ Robustness analysis
```

### Publication

```text
✓ Research paper
✓ Figures
✓ Tables
✓ Supplementary experiments
✓ Reproducible code
```

---

# 31. Important Research Constraints

The project will follow these principles:

### No premature novelty claim

The proposed adaptive mechanism will only be presented as novel after a systematic literature review.

### No single-dataset conclusion

A method should not be declared effective based on one dataset.

### No accuracy-only evaluation

Continual learning must evaluate:

```text
Accuracy
+
Forgetting
+
Transfer
+
Memory
+
Computation
```

### No cherry-picking

Experiments should use fixed protocols and multiple random seeds wherever computationally feasible.

### No comparison without equivalent budgets

Replay-based and non-replay methods should be compared under clearly stated memory and computational constraints.

---

# 32. Potential Research Extensions

Depending on the results and supervisor guidance, the project can later be extended toward:

```text
Continual Learning
       │
       ├── NLP
       │
       ├── Cybersecurity
       │
       ├── Open-World Learning
       │
       ├── Multimodal Learning
       │
       ├── Federated Learning
       │
       └── LLM / Foundation Models
```

This makes the framework extensible rather than tying the entire project to one dataset.

---

# 33. Final Research Goal

The project should ultimately answer:

> **Can a model learn continuously without continuously forgetting?**

More specifically:

```text
             ┌──────────────────────┐
             │   Incoming Knowledge  │
             └──────────┬───────────┘
                        ↓
                Adaptive Learning
                        ↓
       ┌────────────────┴────────────────┐
       ↓                                 ↓
Learn New Information            Preserve Old Knowledge
       ↓                                 ↓
       └────────────────┬────────────────┘
                        ↓
              Efficient Continual Model
                        ↓
          ┌─────────────┼──────────────┐
          ↓             ↓              ↓
       Accuracy      Forgetting     Efficiency
```

The intended contribution is **not merely another implementation of continual learning**. The research objective is to understand and improve the **stability–plasticity–efficiency trade-off** in sequential learning.

---

# 34. Current Project Status

```text
[ ] Literature Review
[ ] Research Gap Identification
[ ] Dataset Selection
[ ] Baseline Framework
[ ] Fine-Tuning Baseline
[ ] Replay Baseline
[ ] Regularization Baseline
[ ] Distillation Baseline
[ ] PEFT Baseline
[ ] Proposed Adaptive Method
[ ] Ablation Study
[ ] Robustness Experiments
[ ] Final Evaluation
[ ] Paper Draft
[ ] Paper Submission
```

---

# 35. Proposed Repository Identity

**Repository name:**

`adaptive-continual-learning`

**Short description:**

> Research framework for adaptive continual learning under non-stationary data, focusing on catastrophic forgetting, knowledge retention, and computational efficiency.

**Research keywords:**

```text
Continual Learning
Deep Learning
Catastrophic Forgetting
Non-Stationary Data
Class-Incremental Learning
Distribution Shift
Knowledge Distillation
Experience Replay
Parameter-Efficient Learning
Adaptive Learning
Stability-Plasticity
```

---

# 36. One-Sentence Proposal

> **We propose to investigate an adaptive continual-learning framework that dynamically balances the retention of previously learned knowledge and adaptation to incoming non-stationary data while considering memory and computational constraints.**
