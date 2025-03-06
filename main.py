#!/usr/bin/env python3
"""
Kontoplan Mapping CLI Tool

A command-line interface for mapping bank transactions to the Danish Standardkontoplan.
This tool combines all the functionality of the various components:
- KontoplanMapper: The core ML-based mapping engine
- EnhancedKontoplanMapper: Extended version with answer file support
- KontoplanTrainingDataGenerator: Generator for synthetic training data

Usage:
    python kontoplan_cli.py [command] [options]

Commands:
    generate-training   Generate synthetic training data and answer file
    train               Train a new model using labeled data
    process             Process bank transactions and map to Kontoplan
    review              Review uncertain predictions and provide feedback
    export              Export data (predictions, history, etc.)
    stats               Show statistics about the mapping system
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path
import datetime
import logging
from tabulate import tabulate

# Import our modules
try:
    # Try to import modules - these imports depend on how you've organized your code
    from KontoplanMapper import KontoplanMapper
    from EnhancedKontoplanMapper import EnhancedKontoplanMapper
    from KontoplanTrainingDataGenerator import KontoplanTrainingDataGenerator
except ImportError as e:
    print(f"Error: Could not import required modules: {str(e)}")
    print("Make sure all the necessary Python files are in the current directory or PYTHONPATH.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kontoplan_cli.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KontoplanCLI")

# Default file paths
DEFAULT_KONTOPLAN_PATH = "2023-01-31-Standardkontoplan.xlsx"
DEFAULT_MODEL_PATH = "kontoplan_mapper_model.joblib"
DEFAULT_ANSWER_FILE_PATH = "transaction_mapping.csv"
DEFAULT_OUTPUT_DIR = "output"

def ensure_output_dir(output_dir=DEFAULT_OUTPUT_DIR):
    """Ensure output directory exists"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_training_command(args):
    """Generate synthetic training data and answer file"""
    print("Generating synthetic training data and answer file...")
    
    # Initialize the generator
    generator = KontoplanTrainingDataGenerator(args.kontoplan)
    
    # Generate transactions
    transactions = generator.generate_transactions(args.num_transactions)
    
    # Create output directory
    output_dir = ensure_output_dir(args.output_dir)
    
    # Output paths
    training_output = os.path.join(output_dir, args.output_name)
    answer_output = os.path.join(output_dir, args.answer_file)
    
    # Save the data
    generator.save_training_data(transactions, training_output)
    
    # Create answer file
    generator.create_answer_file(transactions, answer_output)
    
    print(f"Generated {args.num_transactions} synthetic transactions")
    print(f"Training data saved to {training_output}.csv and {training_output}.xlsx")
    print(f"Answer file saved to {answer_output}")

def train_command(args):
    """Train a new model using labeled data"""
    print(f"Training a new model using {args.training_data}...")
    
    # Load labeled data
    try:
        if args.training_data.endswith('.csv'):
            training_data = pd.read_csv(args.training_data)
        elif args.training_data.endswith('.xlsx'):
            training_data = pd.read_excel(args.training_data)
        else:
            print(f"Unsupported file format: {args.training_data}")
            return
        
        print(f"Loaded {len(training_data)} labeled transactions")
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return
    
    # Initialize the mapper
    mapper = KontoplanMapper()
    
    # Load the Standardkontoplan
    if not mapper.load_kontoplan(args.kontoplan):
        print(f"Failed to load kontoplan from {args.kontoplan}")
        return
    
    # Prepare the training data
    try:
        X_train, X_test, y_train, y_test = mapper.prepare_training_data(training_data)
        print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    except Exception as e:
        print(f"Error preparing training data: {str(e)}")
        return
    
    # Train the model
    print("Training model (this may take a few minutes)...")
    if not mapper.train_model(X_train, y_train):
        print("Failed to train model")
        return
    
    # Evaluate the model
    print("Evaluating model performance...")
    evaluation = mapper.evaluate_model(X_test, y_test)
    
    if evaluation:
        print(f"Model accuracy: {evaluation['accuracy']:.2f}")
        print("\nClassification Report:")
        print(evaluation['classification_report'])
    
    # Save the model
    output_dir = ensure_output_dir(args.output_dir)
    model_output = os.path.join(output_dir, args.model_name)
    
    if mapper.save_model(model_output):
        print(f"Model saved to {model_output}")
    else:
        print(f"Failed to save model to {model_output}")

def process_command(args):
    """Process bank transactions and map to Kontoplan"""
    print(f"Processing transactions from {args.input_file}...")
    
    # Load transactions
    try:
        if args.input_file.endswith('.csv'):
            transactions = pd.read_csv(args.input_file)
        elif args.input_file.endswith('.xlsx'):
            transactions = pd.read_excel(args.input_file)
        else:
            print(f"Unsupported file format: {args.input_file}")
            return
        
        print(f"Loaded {len(transactions)} transactions")
    except Exception as e:
        print(f"Error loading transactions: {str(e)}")
        return
    
    # Initialize the mapper (enhanced if using answer file)
    if args.use_answer_file and os.path.exists(args.answer_file):
        mapper = EnhancedKontoplanMapper()
        mapper.load_answer_file(args.answer_file)
        print(f"Using enhanced mapper with answer file: {args.answer_file}")
    else:
        mapper = KontoplanMapper()
        print("Using standard mapper (no answer file)")
    
    # Load the Standardkontoplan
    if not mapper.load_kontoplan(args.kontoplan):
        print(f"Failed to load kontoplan from {args.kontoplan}")
        return
    
    # Load the model
    if not mapper.load_model(args.model):
        print(f"Failed to load model from {args.model}")
        return
    
    # Process transactions
    try:
        if isinstance(mapper, EnhancedKontoplanMapper):
            predictions = mapper.predict_with_answer_file(transactions)
        else:
            predictions = mapper.predict(transactions)
        
        print(f"Successfully mapped {len(predictions)} transactions")
    except Exception as e:
        print(f"Error processing transactions: {str(e)}")
        return
    
    # Save results
    output_dir = ensure_output_dir(args.output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"mapped_transactions_{timestamp}.xlsx")
    
    try:
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
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
        
        print(f"Results saved to {output_file}")
        
        # Also save uncertain predictions if using enhanced mapper
        if isinstance(mapper, EnhancedKontoplanMapper):
            uncertain = mapper.get_uncertain_predictions()
            if uncertain:
                uncertain_df = pd.DataFrame(uncertain)
                uncertain_file = os.path.join(output_dir, f"uncertain_predictions_{timestamp}.csv")
                uncertain_df.to_csv(uncertain_file, index=False)
                print(f"{len(uncertain)} uncertain predictions saved to {uncertain_file}")
                print("You can review these predictions with the 'review' command")
    
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def review_command(args):
    """Review uncertain predictions and provide feedback"""
    # Check if the file exists
    if not os.path.exists(args.uncertainties_file):
        print(f"File not found: {args.uncertainties_file}")
        return
    
    # Load uncertain predictions
    try:
        uncertain_df = pd.read_csv(args.uncertainties_file)
        print(f"Loaded {len(uncertain_df)} uncertain predictions")
    except Exception as e:
        print(f"Error loading uncertain predictions: {str(e)}")
        return
    
    # Initialize the enhanced mapper
    mapper = EnhancedKontoplanMapper()
    
    # Load the Standardkontoplan
    if not mapper.load_kontoplan(args.kontoplan):
        print(f"Failed to load kontoplan from {args.kontoplan}")
        return
    
    # Load the model
    if not mapper.load_model(args.model):
        print(f"Failed to load model from {args.model}")
        return
    
    # Load the answer file
    if not mapper.load_answer_file(args.answer_file):
        # Create a new answer file if it doesn't exist
        print(f"No existing answer file found at {args.answer_file}. Will create a new one.")
    
    # Interactive review loop
    print("\nStarting interactive review of uncertain predictions.")
    print("For each transaction, provide the correct account number or skip.")
    
    for i, row in uncertain_df.iterrows():
        print("\n" + "="*50)
        print(f"Transaction {i+1}/{len(uncertain_df)}")
        print(f"Description: {row['description']}")
        print(f"Amount: {row['amount']}")
        print(f"Predicted account: {row['predicted_account']} (confidence: {row['confidence']:.2f})")
        
        # Show potential account matches
        accounts_df = mapper.accounts_df
        print("\nPotential account matches:")
        
        # Get top 5 accounts by description similarity
        if accounts_df is not None:
            accounts_df['description_lower'] = accounts_df['description'].str.lower()
            for word in row['description'].lower().split():
                if len(word) > 3:  # Only consider words with more than 3 characters
                    matches = accounts_df[accounts_df['description_lower'].str.contains(word)]
                    if not matches.empty:
                        print("\nAccounts matching keyword: " + word)
                        print(tabulate(matches[['account_number', 'description']], 
                                     headers=['Account', 'Description'], tablefmt='simple'))
        
        # Get user input
        while True:
            user_input = input("\nEnter correct account number (or 's' to skip, 'q' to quit): ")
            
            if user_input.lower() == 's':
                print("Skipping this transaction")
                break
            elif user_input.lower() == 'q':
                print("Exiting review mode")
                return
            else:
                try:
                    account_number = int(user_input)
                    
                    # Check if this is a valid account
                    if accounts_df is not None and account_number not in accounts_df['account_number'].values:
                        print(f"Warning: Account {account_number} not found in kontoplan")
                        confirm = input("Use this account anyway? (y/n): ")
                        if confirm.lower() != 'y':
                            continue
                    
                    # Update the answer file
                    mapper.update_answer_file_with_feedback(row['description'], account_number)
                    print(f"Added mapping: {row['description']} -> {account_number}")
                    break
                except ValueError:
                    print("Invalid input. Please enter a number, 's', or 'q'")
    
    # Save the updated answer file
    output_dir = ensure_output_dir(args.output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"updated_answer_file_{timestamp}.csv")
    
    if mapper.save_answer_file(output_file):
        print(f"\nUpdated answer file saved to {output_file}")
        
        # Also update the original answer file
        mapper.save_answer_file(args.answer_file)
        print(f"Original answer file updated: {args.answer_file}")
    else:
        print(f"Failed to save updated answer file")

def export_command(args):
    """Export data (predictions, history, etc.)"""
    # Initialize the mapper (enhanced version needed for history)
    mapper = EnhancedKontoplanMapper()
    
    # Load the Standardkontoplan
    if not mapper.load_kontoplan(args.kontoplan):
        print(f"Failed to load kontoplan from {args.kontoplan}")
        return
    
    # Load the model
    if not mapper.load_model(args.model):
        print(f"Failed to load model from {args.model}")
        return
    
    # Load the answer file
    if args.answer_file and os.path.exists(args.answer_file):
        mapper.load_answer_file(args.answer_file)
        print(f"Loaded answer file from {args.answer_file}")
    
    # Create output directory
    output_dir = ensure_output_dir(args.output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export kontoplan as CSV
    if args.export_kontoplan:
        if mapper.accounts_df is not None:
            kontoplan_file = os.path.join(output_dir, f"kontoplan_{timestamp}.csv")
            mapper.accounts_df.to_csv(kontoplan_file, index=False)
            print(f"Kontoplan exported to {kontoplan_file}")
    
    # Export transaction history
    if args.export_history:
        history_file = os.path.join(output_dir, f"transaction_history_{timestamp}.csv")
        if mapper.export_transaction_history(history_file):
            print(f"Transaction history exported to {history_file}")
    
    # Export answer file
    if args.export_answer_file:
        if mapper.answer_file_data is not None:
            answer_file = os.path.join(output_dir, f"answer_file_{timestamp}.csv")
            mapper.answer_file_data.to_csv(answer_file, index=False)
            print(f"Answer file exported to {answer_file}")

def stats_command(args):
    """Show statistics about the mapping system"""
    # Initialize the mapper
    if args.use_enhanced and os.path.exists(args.answer_file):
        mapper = EnhancedKontoplanMapper()
        enhanced = True
    else:
        mapper = KontoplanMapper()
        enhanced = False
    
    # Load the Standardkontoplan
    if not mapper.load_kontoplan(args.kontoplan):
        print(f"Failed to load kontoplan from {args.kontoplan}")
        return
    
    # Load the model if it exists
    model_loaded = False
    if os.path.exists(args.model):
        model_loaded = mapper.load_model(args.model)
    
    # Load the answer file if using enhanced mapper
    answer_file_loaded = False
    if enhanced and os.path.exists(args.answer_file):
        answer_file_loaded = mapper.load_answer_file(args.answer_file)
    
    # Print statistics
    print("\n" + "="*50)
    print("Kontoplan Mapping System Statistics")
    print("="*50)
    
    # Kontoplan statistics
    if mapper.accounts_df is not None:
        print(f"\nKontoplan Statistics:")
        print(f"  - Total accounts: {len(mapper.accounts_df)}")
        
        # Distribution by first digit (account type)
        print("\nAccount Distribution:")
        for i in range(1, 10):
            count = len(mapper.accounts_df[mapper.accounts_df['account_number'] // 1000 == i])
            if count > 0:
                print(f"  - {i}XXX accounts: {count}")
    
    # Model statistics
    print(f"\nModel Status: {'Loaded' if model_loaded else 'Not loaded'}")
    if model_loaded and mapper.model is not None:
        if 'classifier' in mapper.model:
            clf = mapper.model['classifier']
            print(f"  - Model type: {type(clf).__name__}")
            print(f"  - Number of estimators: {clf.n_estimators}")
            print(f"  - Max depth: {clf.max_depth}")
            
            if hasattr(clf, 'feature_importances_') and mapper.label_encoder:
                print(f"  - Number of labels: {len(mapper.label_encoder)}")
    
    # Answer file statistics
    if enhanced:
        print(f"\nAnswer File Status: {'Loaded' if answer_file_loaded else 'Not loaded'}")
        if answer_file_loaded and mapper.answer_file_data is not None:
            print(f"  - Number of pattern mappings: {len(mapper.answer_file_data)}")
            
            # Distribution by account
            top_accounts = mapper.answer_file_data['account_number'].value_counts().head(5)
            print("\nTop accounts in answer file:")
            for account, count in top_accounts.items():
                account_desc = "Unknown"
                if mapper.accounts_df is not None:
                    account_matches = mapper.accounts_df[mapper.accounts_df['account_number'] == account]
                    if not account_matches.empty:
                        account_desc = account_matches.iloc[0]['description']
                
                print(f"  - {account} ({account_desc}): {count} patterns")
        
        # Transaction history statistics
        print(f"\nTransaction History:")
        print(f"  - Recorded transactions: {len(mapper.transaction_history)}")
        print(f"  - Uncertain predictions: {len(mapper.uncertain_predictions)}")

def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(
        description='Kontoplan Mapping CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Add global arguments
    parser.add_argument('--kontoplan', default=DEFAULT_KONTOPLAN_PATH,
                        help=f'Path to Standardkontoplan Excel file (default: {DEFAULT_KONTOPLAN_PATH})')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # generate-training command
    generate_parser = subparsers.add_parser('generate-training', 
                                           help='Generate synthetic training data and answer file')
    generate_parser.add_argument('--num-transactions', type=int, default=1000,
                                help='Number of transactions to generate (default: 1000)')
    generate_parser.add_argument('--output-name', default='training_data',
                                help='Base name for output files (default: training_data)')
    generate_parser.add_argument('--answer-file', default=DEFAULT_ANSWER_FILE_PATH,
                                help=f'Path for answer file (default: {DEFAULT_ANSWER_FILE_PATH})')
    generate_parser.set_defaults(func=generate_training_command)
    
    # train command
    train_parser = subparsers.add_parser('train', help='Train a new model using labeled data')
    train_parser.add_argument('--training-data', required=True,
                             help='Path to labeled transaction data (CSV or Excel)')
    train_parser.add_argument('--model-name', default=DEFAULT_MODEL_PATH,
                             help=f'Output model filename (default: {DEFAULT_MODEL_PATH})')
    train_parser.set_defaults(func=train_command)
    
    # process command
    process_parser = subparsers.add_parser('process', 
                                          help='Process bank transactions and map to Kontoplan')
    process_parser.add_argument('--input-file', required=True,
                               help='Path to input transaction file (CSV or Excel)')
    process_parser.add_argument('--model', default=DEFAULT_MODEL_PATH,
                               help=f'Path to model file (default: {DEFAULT_MODEL_PATH})')
    process_parser.add_argument('--use-answer-file', action='store_true',
                               help='Use answer file for improved mapping')
    process_parser.add_argument('--answer-file', default=DEFAULT_ANSWER_FILE_PATH,
                               help=f'Path to answer file (default: {DEFAULT_ANSWER_FILE_PATH})')
    process_parser.set_defaults(func=process_command)
    
    # review command
    review_parser = subparsers.add_parser('review', 
                                         help='Review uncertain predictions and provide feedback')
    review_parser.add_argument('--uncertainties-file', required=True,
                              help='Path to uncertain predictions file (CSV)')
    review_parser.add_argument('--model', default=DEFAULT_MODEL_PATH,
                              help=f'Path to model file (default: {DEFAULT_MODEL_PATH})')
    review_parser.add_argument('--answer-file', default=DEFAULT_ANSWER_FILE_PATH,
                              help=f'Path to answer file (default: {DEFAULT_ANSWER_FILE_PATH})')
    review_parser.set_defaults(func=review_command)
    
    # export command
    export_parser = subparsers.add_parser('export', help='Export data (predictions, history, etc.)')
    export_parser.add_argument('--model', default=DEFAULT_MODEL_PATH,
                              help=f'Path to model file (default: {DEFAULT_MODEL_PATH})')
    export_parser.add_argument('--answer-file', default=DEFAULT_ANSWER_FILE_PATH,
                              help=f'Path to answer file (default: {DEFAULT_ANSWER_FILE_PATH})')
    export_parser.add_argument('--export-kontoplan', action='store_true',
                              help='Export kontoplan to CSV')
    export_parser.add_argument('--export-history', action='store_true',
                              help='Export transaction history to CSV')
    export_parser.add_argument('--export-answer-file', action='store_true',
                              help='Export answer file to CSV')
    export_parser.set_defaults(func=export_command)
    
    # stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics about the mapping system')
    stats_parser.add_argument('--model', default=DEFAULT_MODEL_PATH,
                             help=f'Path to model file (default: {DEFAULT_MODEL_PATH})')
    stats_parser.add_argument('--use-enhanced', action='store_true',
                             help='Use enhanced mapper with answer file support')
    stats_parser.add_argument('--answer-file', default=DEFAULT_ANSWER_FILE_PATH,
                             help=f'Path to answer file (default: {DEFAULT_ANSWER_FILE_PATH})')
    stats_parser.set_defaults(func=stats_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the selected command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()