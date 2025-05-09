# Enhancing Data Quality

Enhancing Data Quality in Large-Scale Data Warehouses Using AI: A Case Study on Istanbul Shopping Mall Transactions

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example Workflow](#example-workflow)
- [Implementation Details](#implementation-details)
- [Output Examples](#output-examples)
- [Research Background](#research-background)
- [License](#license)

## Overview

This tool implements a systematic approach to addressing data quality challenges in large-scale data warehouses through machine learning techniques. It can detect anomalies, handle missing values intelligently, and provide comprehensive reports on data quality metrics.

The implementation follows the methodology outlined in the research paper, combining Isolation Forest for anomaly detection, k-Nearest Neighbors (KNN) for intelligent imputation, and standardization to create a comprehensive data quality monitoring framework.

## Features

- **Initial Data Quality Assessment**
  - Detection of missing values with percentage analysis
  - Identification of duplicate entries
  - Data type and distribution analysis
  - Visualization of missing data patterns

- **Automated Data Preprocessing**
  - Date format standardization
  - Basic handling of missing values
  - Duplicate removal

- **Anomaly Detection**
  - Implementation of Isolation Forest algorithm
  - Detection of unusual patterns in numerical data
  - Visual representation of identified anomalies

- **Intelligent Imputation**
  - KNN-based imputation for missing values
  - Context-aware value estimation preserving relationships between variables

- **Data Standardization**
  - Transformation of numerical features to have zero mean and unit variance
  - Ensuring feature comparability regardless of original scales

- **Comprehensive Reporting**
  - Visual and statistical summaries of data quality issues
  - Before-and-after comparisons
  - Category-wise anomaly analysis

## Installation

### Prerequisites
- Python 3.7 or higher

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/461117219/enhancing_data_quality.git
   cd enhancing_data_quality
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
```python
from enhanced_ai_data_quality_tool import DataQualityMonitor

# Initialize the monitor
monitor = DataQualityMonitor(contamination=0.01, n_neighbors=5)

# Load data
df = monitor.load_data('your_data.csv')

# Run the full monitoring pipeline
monitor.initial_assessment()
monitor.preprocess_data()
monitor.detect_anomalies(['feature1', 'feature2', 'feature3'])
monitor.apply_knn_imputation()
monitor.standardize_features()
monitor.visualize_anomalies('feature1', 'feature2')
report = monitor.generate_report()
```

### Configuration Parameters

- `contamination`: Expected proportion of anomalies in the dataset (default: 0.01)
- `n_neighbors`: Number of neighbors to use for KNN imputation (default: 5)

## Example Workflow

Below is a step-by-step walkthrough of how to use this tool for data quality monitoring:

```python
# Initialize the monitor
monitor = DataQualityMonitor()

# Load the retail shopping data
df = monitor.load_data('shopping_data.csv')

# Perform initial quality assessment
monitor.initial_assessment()

# Preprocess the data
monitor.preprocess_data()

# Define numerical features for analysis
numeric_features = ['age', 'quantity', 'price']

# Detect anomalies in the specified features
anomalies = monitor.detect_anomalies(numeric_features)

# Apply KNN imputation to refine missing values
monitor.apply_knn_imputation(numeric_features)

# Standardize the numerical features
monitor.standardize_features(numeric_features)

# Visualize anomalies in quantity vs price
monitor.visualize_anomalies('quantity', 'price')

# Generate a comprehensive data quality report
report = monitor.generate_report()
```

## Implementation Details

### Core Components

1. **DataQualityMonitor Class**
   The main class providing all data quality monitoring functionality.

2. **Isolation Forest for Anomaly Detection**
   Unsupervised machine learning algorithm that efficiently identifies anomalies by isolating observations.

3. **KNN Imputation**
   Intelligent method for handling missing values by finding similar records and using their values for estimation.

4. **Data Visualization**
   Components to visualize data quality issues and the impact of corrections.

### Key Methods

- `load_data(file_path)`: Load data from a CSV file
- `initial_assessment()`: Perform initial data quality assessment
- `preprocess_data()`: Apply basic preprocessing
- `detect_anomalies(numeric_features)`: Detect anomalies in numerical features
- `apply_knn_imputation(numeric_features)`: Apply KNN imputation for missing values
- `standardize_features(numeric_features)`: Standardize numerical features
- `visualize_anomalies(x_feature, y_feature)`: Visualize detected anomalies
- `generate_report()`: Generate a comprehensive data quality report

## Output Examples


### Console Output

The tool provides detailed console output including:
- Summary statistics
- Missing value percentages
- Anomaly detection results
- Data quality improvement metrics
