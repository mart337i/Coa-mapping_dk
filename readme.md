# Bank Transaction to Danish Kontoplan Mapper

This system uses machine learning to automatically map bank transactions to the appropriate account numbers in the Danish Standard Chart of Accounts (Standardkontoplan).

## Overview

The Kontoplan Mapper system consists of several integrated components:

1. **Core Mapping Engine**: ML-based system that categorizes bank transactions
2. **Training Data Generator**: Creates synthetic data for initial training
3. **Enhanced Mapper**: Extends core functionality with an answer file for continuous improvement
4. **Command-line Interface**: Unified interface for all components
5. **Configuration Manager**: Flexible configuration system

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - joblib
  - pyyaml
  - xlsxwriter
  - tabulate
  - faker (for generating synthetic data)

### Install Required Packages

```bash
pip install pandas numpy scikit-learn joblib pyyaml xlsxwriter tabulate faker
```

### Download the Code

Clone the repository or download all Python files to your local machine.

## Quick Start

### 1. Generate Training Data

Start by generating synthetic training data to train the initial model:

```bash
python kontoplan_cli.py generate-training --kontoplan 2023-01-31-Standardkontoplan.xlsx --num-transactions 2000
```

This will create:

- `output/training_data.csv`: Training data in CSV format
- `output/training_data.xlsx`: Training data in Excel format
- `transaction_mapping.csv`: Answer file with correct mappings

### 2. Train the Model

Use the generated training data to train the machine learning model:

```bash
python kontoplan_cli.py train --training-data output/training_data.csv
```

This will create:

- `output/kontoplan_mapper_model.joblib`: The trained model file

### 3. Process Bank Transactions

Now you can process your actual bank transactions:

```bash
python kontoplan_cli.py process --input-file your_bank_transactions.csv --use-answer-file
```

This will produce:

- `output/mapped_transactions_[timestamp].xlsx`: Transactions with predicted account numbers
- `output/uncertain_predictions_[timestamp].csv`: Transactions with low confidence predictions

### 4. Review Uncertain Predictions

Review and correct uncertain predictions to improve future mapping:

```bash
python kontoplan_cli.py review --uncertainties-file output/uncertain_predictions_[timestamp].csv
```

## Components in Detail

### 1. Core Mapping Engine (`dk_kontoplan_mapping.py`)

The core engine that learns to map transactions to accounts using:

- Text analysis of transaction descriptions
- Amount-based features (positive/negative, magnitude)
- Date-based features (day of week, month, etc. if available)

### 2. Training Data Generator (`training_data_generator.py`)

Creates synthetic but realistic Danish bank transactions with:

- Proper transaction descriptions for different account types
- Realistic amounts and date patterns
- Labels for each transaction based on the Danish Standardkontoplan

### 3. Enhanced Mapper (`enhanced_mapper.py`)

Extends the core mapper with:

- Answer file integration for higher accuracy
- Confidence scores for predictions
- Tracking of uncertain predictions
- Continuous learning from user feedback

### 4. Command-line Interface (`kontoplan_cli.py`)

Unified command-line interface with multiple commands:

- `generate-training`: Create synthetic training data
- `train`: Train the ML model
- `process`: Map bank transactions to account numbers
- `review`: Review and correct uncertain predictions
- `export`: Export data for external analysis
- `stats`: Show system statistics

### 5. Configuration Manager (`kontoplan_config.py`)

Flexible configuration system that:

- Loads settings from YAML files
- Supports environment variables
- Integrates with command-line arguments
- Provides sensible defaults

## Using the Answer File

The answer file is a key component for improving accuracy over time. It contains:

- Transaction patterns that have been mapped to specific accounts
- Keywords that help identify similar transactions
- User-verified mappings from the review process

The Enhanced Mapper uses the answer file first for exact or close matches, then falls back to the ML model for new patterns.

## Customization

### Configuration

Create a default configuration file:

```bash
python kontoplan_config.py create-default --output myconfig.yml
```

Edit this file to customize settings, then use it with any command:

```bash
python kontoplan_cli.py process --config myconfig.yml --input-file transactions.csv
```

### Training with Your Own Data

If you have already-labeled transactions, you can use them directly:

1. Prepare a CSV/Excel file with columns:

   - `description`: Transaction text
   - `amount`: Transaction amount
   - `account_number`: The correct account number in the Standardkontoplan
   - Optional: `date` in a standard format
2. Train using your file:

```bash
python kontoplan_cli.py train --training-data your_labeled_data.csv
```

## Continuous Improvement

For best results:

1. Always use the `--use-answer-file` option when processing transactions
2. Regularly review uncertain predictions with the `review` command
3. Periodically examine the system statistics with the `stats` command

The system learns from your corrections and improves accuracy over time.

## Input File Formats

The system supports various bank transaction formats:

- **CSV files**: Standard CSV format with transaction data
- **Excel files**: Excel format with transaction data
- **Custom formats**: The system attempts to detect column names automatically

Minimum required columns:

- Transaction description (will detect columns named: description, text, narrative, details, memo, note)
- Amount (will detect columns named: amount, sum, total, beløb, værdi)

## Troubleshooting

### Common Issues

1. **Model not found**:

   - Make sure you've run the `train` command first
   - Check the model path in configuration or command-line arguments
2. **Low accuracy**:

   - Generate more training data with `--num-transactions 5000`
   - Review more uncertain predictions to build a better answer file
3. **File format errors**:

   - Ensure your bank transaction file has the required columns
