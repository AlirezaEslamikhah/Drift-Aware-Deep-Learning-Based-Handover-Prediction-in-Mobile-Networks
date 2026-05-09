I based this README on the uploaded report, including the drift-aware workflow, drift-injection methodology, and reported LSTM degradation results. 

````markdown
# Drift-Aware Deep Learning-Based Handover Prediction in Mobile Networks

This repository contains the implementation and experimental framework for **drift-aware handover prediction in mobile networks** using deep learning and RSRP time-series data.

The project investigates how short-term feature drift affects an LSTM-based handover prediction pipeline and evaluates controlled **sudden**, **gradual**, and **incremental** drift scenarios on real-network radio measurements.

---

## Overview

Modern 5G and beyond mobile networks rely on efficient handover mechanisms to maintain service continuity as user equipment moves between cells. Deep learning models, especially LSTMs, can improve handover prediction by forecasting future radio measurements such as **Reference Signal Received Power (RSRP)**.

However, real radio environments are not always stationary. Temporary changes caused by blockage, shadowing, traffic concentration, mobility pattern changes, or local events can distort measurement streams and reduce prediction reliability.

This project evaluates the impact of such feature drift on an LSTM-based RSRP predictor and motivates a drift-aware online monitoring and adaptation pipeline.

---

## Key Features

- LSTM-based RSRP forecasting for handover prediction support
- Controlled concept-drift injection into real-network RSRP streams
- Drift scenarios:
  - Sudden drift
  - Gradual drift
  - Incremental drift
- Drift characterization using:
  - Wasserstein distance
  - Kolmogorov–Smirnov statistic
  - Population Stability Index
  - Latent center shift
  - MAD flag rate
- Prediction degradation analysis using:
  - MAE
  - RMSE
  - R²
  - Prediction-to-target correlation
  - Percentage of predictions within 3 dB
- Proposed online drift-detection design using ADWIN on the prediction-error stream
- Proposed soft-forgetting adaptation strategy with a state-machine controller

---

## Methodology

The framework is organized into four main stages:

```text
Windowed RSRP Stream
        |
        v
Controlled Drift Injection
        |
        v
Baseline LSTM Forecasting
        |
        v
Prediction Error Monitoring
        |
        v
Drift Detection and Adaptation
````

### 1. Controlled Drift Generation

The original RSRP sequence is divided into non-overlapping windows of 10 samples. For each window, the following statistics are extracted:

* Mean
* Standard deviation
* Delta
* Slope

Windows are then grouped into macro-clusters based on signal strength and further divided into micro-clusters according to trend and volatility. These clusters are used as donor pools to generate controlled drift scenarios.

### 2. Drift Impact Evaluation

The drifted test streams are passed through the baseline LSTM model. Prediction quality is compared against normal test conditions to quantify how much each drift profile degrades performance.

### 3. Online Drift Detection

Instead of detecting drift directly on highly overlapping LSTM input windows, the proposed framework monitors the prediction-error stream. ADWIN is used to detect statistically meaningful changes in error behavior.

### 4. Soft-Forgetting Adaptation

A future adaptation module is proposed using a four-state controller:

* `Normal`
* `Warning`
* `Adaptation`
* `Recovery`

This design allows the model to adapt to recent post-drift samples while preserving useful long-term temporal knowledge.

---

## Experimental Results

The baseline LSTM performs well under normal conditions, but all injected drift scenarios reduce prediction accuracy.

| Scenario          |   MAE |  RMSE |    R² | Correlation | ≤ 3 dB |
| ----------------- | ----: | ----: | ----: | ----------: | -----: |
| Normal            | 1.861 | 2.615 | 0.858 |       0.931 |  0.808 |
| Gradual Drift     | 3.442 | 5.190 | 0.694 |       0.837 |  0.601 |
| Incremental Drift | 2.456 | 3.636 | 0.736 |       0.858 |  0.713 |
| Sudden Drift      | 3.834 | 5.827 | 0.650 |       0.814 |  0.555 |

### Main Finding

Sudden drift causes the largest degradation, increasing:

* MAE from `1.8607` to `3.8344`
* RMSE from `2.6154` to `5.8272`

Incremental drift has the mildest effect, while gradual drift produces an intermediate but still substantial performance loss.

---

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── drifted/
├── notebooks/
│   ├── drift_generation.ipynb
│   ├── lstm_training.ipynb
│   └── drift_evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── drift_generator.py
│   ├── lstm_model.py
│   ├── evaluation.py
│   ├── drift_detection.py
│   └── adaptation.py
├── results/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/drift-aware-handover-prediction.git
cd drift-aware-handover-prediction
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Example Requirements

```text
numpy
pandas
scikit-learn
matplotlib
tensorflow
river
scipy
```

---

## Usage

### 1. Preprocess the RSRP Dataset

```bash
python src/data_preprocessing.py
```

### 2. Generate Drift Scenarios

```bash
python src/drift_generator.py --scenario sudden
python src/drift_generator.py --scenario gradual
python src/drift_generator.py --scenario incremental
```

### 3. Train the Baseline LSTM

```bash
python src/lstm_model.py --mode train
```

### 4. Evaluate Under Normal and Drifted Conditions

```bash
python src/evaluation.py --scenario normal
python src/evaluation.py --scenario sudden
python src/evaluation.py --scenario gradual
python src/evaluation.py --scenario incremental
```

### 5. Run Drift Detection

```bash
python src/drift_detection.py --method adwin
```

---

## Dataset

The evaluation uses a real-network RSRP dataset collected during a drive-test campaign in Belo Horizonte, Brazil. The dataset was originally used in prior deep-learning-based handover prediction research and contains radio measurements suitable for forecasting future RSRP behavior.

Due to dataset licensing or availability constraints, the raw dataset may need to be downloaded separately or placed manually under:

```text
data/raw/
```

---

## Metrics

The following metrics are used to evaluate prediction quality:

| Metric      | Description                                             |
| ----------- | ------------------------------------------------------- |
| MAE         | Mean Absolute Error between predicted and true RSRP     |
| RMSE        | Root Mean Square Error                                  |
| R²          | Coefficient of determination                            |
| Correlation | Prediction-to-target correlation                        |
| ≤ 3 dB      | Fraction of predictions within 3 dB of the ground truth |

---

## Drift Scenarios

### Sudden Drift

The signal stream abruptly shifts from normal behavior to strong-drift donor samples.

### Gradual Drift

The stream progressively transitions from normal behavior to mild drift and then to strong drift.

### Incremental Drift

The original signal is increasingly blended with strong-drift samples, producing a smoother and milder perturbation.

---

## Future Work

Planned extensions include:

* Full online implementation of ADWIN-based drift detection
* Soft-forgetting LSTM adaptation after detected drift
* Integration with a downstream handover-decision classifier
* Comparison of offline classifiers such as:

  * Random Forest
  * Support Vector Machine
  * Multilayer Perceptron
* Comparison with adaptive stream-learning classifiers such as:

  * Adaptive Random Forest
  * Hoeffding-based classifiers
* End-to-end drift-aware handover prediction pipeline for dynamic mobile environments

---

## Citation

If you use this project, please cite the related paper:

```bibtex
@article{eslamikhah2025driftaware,
  title={Drift-Aware Deep Learning-Based Handover Prediction in Mobile Networks},
  author={Eslamikhah, Alireza},
  institution={Department of Computer Science, University of Regina},
  year={2025}
}
```

---

## Author

**Alireza Eslamikhah**
Department of Computer Science
University of Regina
Regina, Saskatchewan, Canada

---

## License

This project is intended for academic and research use. Add an appropriate license before public release, such as MIT, Apache-2.0, or GPL-3.0.

```
```
