#!/usr/bin/env python3
"""
Complete Example: Bank Transaction to Danish Kontoplan Mapper

This script demonstrates a complete workflow of the Kontoplan Mapper system:
1. Generate synthetic training data
2. Train the machine learning model
3. Generate sample bank transactions
4. Process the transactions using the trained model
5. Review uncertain predictions
6. Show system statistics

Run this script to see the entire system in action.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import datetime
from pathlib import Path

# Ensure that we're in the correct directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Try to import our modules
try:
    from KontoplanTrainingDataGenerator import KontoplanTrainingDataGenerator
    from KontoplanMapper import KontoplanMapper
    from EnhancedKontoplanMapper import EnhancedKontoplanMapper
    from ConfigManager import ConfigManager
except ImportError as e:
    print(f"Error: Could not import required modules: {str(e)}")
    print("Make sure all the necessary Python files are in the current directory or PYTHONPATH.")
    sys.exit(1)


def create_sample_bank_file(output_path, num_transactions=50):
    """Create a sample bank transaction file"""
    # Bank descriptions for common transaction types
    description_patterns = [
        "BETALING TIL {recipient}",
        "OVERFØRSEL TIL {recipient}",
        "INDBETALING FRA {sender}",
        "MOBILEPAY FRA {sender}",
        "BETALING MED KORT: {location}",
        "LØNSYSTEM: {company}",
        "UDBYTTE: {company}",
        "FORSIKRING: {company}",
        "ABONNEMENT: {service}",
        "KONTANTHÆVNING: {location}"
    ]
    
    # Lists for substitution
    recipients = [
        "UDLEJER A/S",
        "ELSELSKAB",
        "VANDSELSKAB",
        "FORSIKRING A/S",
        "PENSIONSSELSKAB",
        "REVISOR",
        "ADVOKAT",
        "KONSULENT",
        "LEVERANDØR A/S",
        "TRANSPORT A/S"
    ]
    
    senders = [
        "KUNDE A/S",
        "NIELSEN APS",
        "KLIENT",
        "SAMARBEJDSPARTNER",
        "HANSEN OG SØNER",
        "UNDERLEVERANDØR",
        "DISTRIBUTØR",
        "FORHANDLER A/S",
        "BUTIK",
        "ONLINE SHOP"
    ]
    
    locations = [
        "SUPERMARKET",
        "TANKSTATION",
        "BYGGEMARKED",
        "RESTAURANT",
        "HARDWARE BUTIK",
        "KONTORUDSTYR A/S",
        "HOTEL",
        "FLYSELSKAB",
        "TAXASELSKAB",
        "BILUDLEJNING"
    ]
    
    companies = [
        "FIRMA A/S",
        "KONCERN APS",
        "HOLDING A/S",
        "PARTNER APS",
        "INTERNATIONAL LTD",
        "GROUP APS",
        "INVEST A/S",
        "INDUSTRI",
        "ONLINE SERVICES",
        "HANSEN & CO"
    ]
    
    services = [
        "MOBIL TELEFONI",
        "INTERNET",
        "STREAMING",
        "SOFTWARELICENS",
        "CRM SYSTEM",
        "ERP SYSTEM",
        "CLOUD STORAGE",
        "ANTIVIRUS",
        "MARKETING TOOLS",
        "REGNSKABSSYSTEM"
    ]
    
    # Generate transactions
    transactions = []
    start_date = datetime.datetime.now() - datetime.timedelta(days=180)
    
    for i in range(num_transactions):
        # Select random pattern and substitutions
        pattern = np.random.choice(description_patterns)
        recipient = np.random.choice(recipients)
        sender = np.random.choice(senders)
        location = np.random.choice(locations)
        company = np.random.choice(companies)
        service = np.random.choice(services)
        
        # Create description
        description = pattern.format(
            recipient=recipient,
            sender=sender,
            location=location,
            company=company,
            service=service
        )
        
        # Determine if it's an expense or income
        is_expense = "TIL" in pattern or "MED KORT" in pattern or "HÆVNING" in pattern or "ABONNEMENT" in pattern or "FORSIKRING" in pattern
        
        # Generate amount (negative for expenses, positive for income)
        if is_expense:
            amount = -np.random.uniform(100, 10000)
        else:
            amount = np.random.uniform(1000, 50000)
        
        # Generate date
        days_offset = np.random.randint(0, 180)
        date = start_date + datetime.timedelta(days=days_offset)
        
        # Add transaction
        transactions.append({
            'date': date.strftime('%Y-%m-%d'),
            'description': description,
            'amount': round(amount, 2)
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(transactions)
    df = df.sort_values('date')
    df.to_csv(output_path, index=False)
    
    return output_path


def run_complete_example(args):
    """Run the complete example workflow"""
    print("\n" + "="*80)
    print("COMPLETE EXAMPLE: BANK TRANSACTION TO DANISH KONTOPLAN MAPPER")
    print("="*80 + "\n")
    
    # Initialize the configuration manager
    config = ConfigManager()
    
    # Create output directory
    output_dir = config.get('paths', 'output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    kontoplan_path = config.get('paths', 'kontoplan', '2023-01-31-Standardkontoplan.xlsx')
    model_path = os.path.join(output_dir, 'example_model.joblib')
    answer_file_path = os.path.join(output_dir, 'example_answer_file.csv')
    training_data_path = os.path.join(output_dir, 'example_training_data')
    sample_bank_file = os.path.join(output_dir, 'example_bank_transactions.csv')
    mapped_output_file = os.path.join(output_dir, 'example_mapped_transactions.xlsx')
    uncertain_file = os.path.join(output_dir, 'example_uncertain_predictions.csv')
    
    # Step 1: Generate training data
    print("\nSTEP 1: Generating synthetic training data...\n")
    
    generator = KontoplanTrainingDataGenerator(kontoplan_path)
    training_data = generator.generate_transactions(num_transactions=1000)
    generator.save_training_data(training_data, training_data_path)
    generator.create_answer_file(training_data, answer_file_path)
    
    print(f"Generated 1000 synthetic transactions")
    print(f"Training data saved to {training_data_path}.csv and {training_data_path}.xlsx")
    print(f"Answer file saved to {answer_file_path}")
    
    # Step 2: Train the model
    print("\nSTEP 2: Training the machine learning model...\n")
    
    mapper = KontoplanMapper()
    mapper.load_kontoplan(kontoplan_path)
    
    X_train, X_test, y_train, y_test = mapper.prepare_training_data(training_data)
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    print("Training model (this may take a few minutes)...")
    mapper.train_model(X_train, y_train)
    
    print("Evaluating model performance...")
    evaluation = mapper.evaluate_model(X_test, y_test)
    
    if evaluation:
        print(f"Model accuracy: {evaluation['accuracy']:.2f}")
        print("\nClassification Report:")
        print(evaluation['classification_report'])
    
    print(f"Saving model to {model_path}...")
    mapper.save_model(model_path)
    
    # Step 3: Generate sample bank transactions
    print("\nSTEP 3: Generating sample bank transactions...\n")
    
    create_sample_bank_file(sample_bank_file, num_transactions=50)
    print(f"Created sample bank file with 50 transactions: {sample_bank_file}")
    
    # Step 4: Process the transactions
    print("\nSTEP 4: Processing bank transactions...\n")
    
    enhanced_mapper = EnhancedKontoplanMapper()
    enhanced_mapper.load_kontoplan(kontoplan_path)
    enhanced_mapper.load_model(model_path)
    enhanced_mapper.load_answer_file(answer_file_path)
    
    bank_data = pd.read_csv(sample_bank_file)
    predictions = enhanced_mapper.predict_with_answer_file(bank_data)
    
    # Show some predictions
    print("\nSample predictions:")
    for i, row in predictions.head(5).iterrows():
        print(f"{row['description'][:40]}... | Amount: {row['amount']} | " 
              f"Predicted account: {row['predicted_account']} | "
              f"Confidence: {row['confidence']:.2f} | "
              f"Source: {row['prediction_source']}")
    
    # Save predictions
    with pd.ExcelWriter(mapped_output_file, engine='xlsxwriter') as writer:
        # Main transactions sheet
        predictions.to_excel(writer, sheet_name='Mapped Transactions', index=False)
        
        # Summary by account
        account_summary = predictions.groupby('predicted_account').agg({
            'amount': ['sum', 'count'],
            'confidence': 'mean'
        })
        account_summary.columns = ['Total Amount', 'Transaction Count', 'Avg Confidence']
        account_summary.reset_index().to_excel(writer, sheet_name='Account Summary', index=False)
        
        # Low confidence transactions
        low_conf = predictions[predictions['confidence'] < 0.6].sort_values('confidence')
        if len(low_conf) > 0:
            low_conf.to_excel(writer, sheet_name='Low Confidence', index=False)
    
    print(f"\nFull predictions saved to {mapped_output_file}")
    
    # Save uncertain predictions
    uncertain = enhanced_mapper.get_uncertain_predictions()
    if uncertain:
        uncertain_df = pd.DataFrame(uncertain)
        uncertain_df.to_csv(uncertain_file, index=False)
        print(f"{len(uncertain)} uncertain predictions saved to {uncertain_file}")
    
    # Step 5: Show system statistics
    print("\nSTEP 5: System statistics...\n")
    
    # Kontoplan statistics
    print(f"Kontoplan Statistics:")
    print(f"  - Total accounts: {len(enhanced_mapper.accounts_df)}")
    
    # Model statistics
    print(f"\nModel Status: Model loaded")
    if enhanced_mapper.model is not None:
        if 'classifier' in enhanced_mapper.model:
            clf = enhanced_mapper.model['classifier']
            print(f"  - Model type: {type(clf).__name__}")
            print(f"  - Number of estimators: {clf.n_estimators}")
            print(f"  - Max depth: {clf.max_depth}")
            
            if hasattr(clf, 'feature_importances_') and enhanced_mapper.label_encoder:
                print(f"  - Number of labels: {len(enhanced_mapper.label_encoder)}")
    
    # Answer file statistics
    print(f"\nAnswer File Status: Loaded")
    if enhanced_mapper.answer_file_data is not None:
        print(f"  - Number of pattern mappings: {len(enhanced_mapper.answer_file_data)}")
    
    # Print conclusion
    print("\n" + "="*80)
    print("EXAMPLE WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nAll files have been saved to the '{output_dir}' directory.")
    print("""
Next steps you can take:
1. Examine the mapped transactions in the Excel file
2. Review uncertain predictions
3. Use the full CLI tool for more options:
   python kontoplan_cli.py --help
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a complete example of the Kontoplan Mapper system")
    parser.add_argument('--output-dir', default='output', help='Output directory for example files')
    parser.add_argument('--kontoplan', default='20230131Standardkontoplan.xlsx', 
                      help='Path to the Standardkontoplan Excel file')
    
    args = parser.parse_args()
    
    # Run the complete example
    run_complete_example(args)