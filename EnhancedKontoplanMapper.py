import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import re
import logging
from datetime import datetime

# Import the base mapper class
from KontoplanMapper import KontoplanMapper

class EnhancedKontoplanMapper(KontoplanMapper):
    """
    Enhanced version of KontoplanMapper that can learn from an answer file
    and continuously improve its predictions
    """
    
    def __init__(self):
        super().__init__()
        self.answer_file_data = None
        self.pattern_mapping = {}
        self.transaction_history = []
        self.uncertain_predictions = []
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("kontoplan_mapper.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("EnhancedKontoplanMapper")
    
    def load_answer_file(self, file_path):
        """
        Load the answer file containing transaction pattern mappings
        
        Parameters:
        ----------
        file_path : str
            Path to the answer file (CSV)
            
        Returns:
        -------
        bool
            True if loading was successful
        """
        try:
            self.answer_file_data = pd.read_csv(file_path)
            self.logger.info(f"Loaded answer file with {len(self.answer_file_data)} pattern mappings")
            
            # Create pattern mapping dictionary
            for _, row in self.answer_file_data.iterrows():
                pattern = str(row['pattern']).lower()
                self.pattern_mapping[pattern] = {
                    'account_number': row['account_number'],
                    'keywords': str(row['keywords']).lower().split()
                }
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading answer file: {str(e)}")
            return False
    
    def _find_best_pattern_match(self, description):
        """
        Find the best matching pattern in the answer file
        
        Parameters:
        ----------
        description : str
            Transaction description to match
            
        Returns:
        -------
        tuple
            (account_number, confidence)
        """
        if not self.pattern_mapping:
            return None, 0.0
            
        description = description.lower()
        
        # Check for exact matches first
        if description in self.pattern_mapping:
            return self.pattern_mapping[description]['account_number'], 1.0
        
        # Extract words from the description
        words = set(re.findall(r'\w+', description))
        
        best_match = None
        best_score = 0.0
        
        # Find the pattern with the highest keyword match
        for pattern, data in self.pattern_mapping.items():
            # Calculate the number of matching keywords
            keywords = set(data['keywords'])
            match_count = len(words.intersection(keywords))
            
            if match_count > 0:
                # Calculate a score based on the proportion of matching keywords
                keyword_score = match_count / len(keywords)
                
                # Enhance score with Jaccard similarity between descriptions
                pattern_words = set(re.findall(r'\w+', pattern))
                jaccard = len(words.intersection(pattern_words)) / len(words.union(pattern_words))
                
                # Combined score (weighted average)
                score = 0.7 * keyword_score + 0.3 * jaccard
                
                if score > best_score:
                    best_score = score
                    best_match = data['account_number']
        
        return best_match, best_score
    
    def predict_with_answer_file(self, transactions):
        """
        Predict account categories using both the ML model and answer file
        
        Parameters:
        ----------
        transactions : pandas.DataFrame
            DataFrame with bank transactions
            
        Returns:
        -------
        pandas.DataFrame
            Transactions with predicted account categories
        """
        # First get ML model predictions
        df = super().predict(transactions)
        
        # Then enhance with answer file pattern matching
        for i, row in df.iterrows():
            description = row['description']
            
            # Try to find a match in the answer file
            account_number, confidence = self._find_best_pattern_match(description)
            
            if account_number is not None and confidence > row['confidence']:
                # Answer file has a better match
                df.at[i, 'predicted_account'] = account_number
                df.at[i, 'confidence'] = confidence
                df.at[i, 'prediction_source'] = 'answer_file'
            else:
                # Keep the ML model prediction
                df.at[i, 'prediction_source'] = 'ml_model'
                
            # Add to transaction history for continuous learning
            self.transaction_history.append({
                'description': description,
                'amount': row['amount'],
                'predicted_account': df.at[i, 'predicted_account'],
                'confidence': df.at[i, 'confidence'],
                'timestamp': datetime.now()
            })
            
            # Flag uncertain predictions for review
            if df.at[i, 'confidence'] < 0.6:
                self.uncertain_predictions.append({
                    'description': description,
                    'amount': row['amount'],
                    'predicted_account': df.at[i, 'predicted_account'],
                    'confidence': df.at[i, 'confidence']
                })
        
        return df
    
    def update_answer_file_with_feedback(self, transaction_description, correct_account):
        """
        Update the answer file with user feedback on a prediction
        
        Parameters:
        ----------
        transaction_description : str
            The transaction description
        correct_account : int
            The correct account number
            
        Returns:
        -------
        bool
            True if the update was successful
        """
        try:
            description = transaction_description.lower()
            
            # Extract keywords from the description
            words = re.findall(r'\w+', description)
            keywords = ' '.join(sorted(set(words), key=words.count, reverse=True)[:5])
            
            # Create new mapping entry
            new_entry = pd.DataFrame([{
                'pattern': description,
                'keywords': keywords,
                'account_number': correct_account,
                'account_description': self.accounts_df.loc[
                    self.accounts_df['account_number'] == correct_account, 
                    'description'
                ].values[0] if self.accounts_df is not None else ''
            }])
            
            # Update pattern mapping
            self.pattern_mapping[description] = {
                'account_number': correct_account,
                'keywords': keywords.split()
            }
            
            # Add to answer file data
            if self.answer_file_data is None:
                self.answer_file_data = new_entry
            else:
                # Check if this pattern already exists
                pattern_exists = (self.answer_file_data['pattern'] == description).any()
                
                if pattern_exists:
                    # Update existing entry
                    self.answer_file_data.loc[
                        self.answer_file_data['pattern'] == description, 
                        'account_number'
                    ] = correct_account
                else:
                    # Add new entry
                    self.answer_file_data = pd.concat([self.answer_file_data, new_entry], ignore_index=True)
            
            self.logger.info(f"Added feedback for transaction: {description} -> {correct_account}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating answer file with feedback: {str(e)}")
            return False
    
    def save_answer_file(self, file_path):
        """
        Save the updated answer file
        
        Parameters:
        ----------
        file_path : str
            Path to save the answer file
            
        Returns:
        -------
        bool
            True if saving was successful
        """
        try:
            if self.answer_file_data is not None:
                self.answer_file_data.to_csv(file_path, index=False)
                self.logger.info(f"Saved answer file with {len(self.answer_file_data)} pattern mappings")
                return True
            else:
                self.logger.error("No answer file data to save")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving answer file: {str(e)}")
            return False
    
    def get_uncertain_predictions(self, confidence_threshold=0.6):
        """
        Get a list of uncertain predictions that need review
        
        Parameters:
        ----------
        confidence_threshold : float
            Confidence threshold below which predictions are considered uncertain
            
        Returns:
        -------
        list
            List of uncertain prediction dictionaries
        """
        return [p for p in self.uncertain_predictions if p['confidence'] < confidence_threshold]
    
    def retrain_with_transaction_history(self, min_history_size=100):
        """
        Retrain the model using transaction history
        
        Parameters:
        ----------
        min_history_size : int
            Minimum number of transactions needed for retraining
            
        Returns:
        -------
        bool
            True if retraining was successful
        """
        if len(self.transaction_history) < min_history_size:
            self.logger.info(f"Not enough transaction history for retraining ({len(self.transaction_history)}/{min_history_size})")
            return False
            
        try:
            # Create a dataframe from transaction history
            history_df = pd.DataFrame(self.transaction_history)
            
            # Keep only high-confidence predictions
            training_data = history_df[history_df['confidence'] > 0.8].copy()
            
            # Add required columns for model training
            training_data['date'] = pd.to_datetime(training_data['timestamp'])
            
            # Prepare training data
            X_train, X_test, y_train, y_test = self.prepare_training_data(training_data)
            
            # Train the model
            success = self.train_model(X_train, y_train)
            
            if success:
                # Evaluate the model
                evaluation = self.evaluate_model(X_test, y_test)
                self.logger.info(f"Model retrained with {len(training_data)} transactions. New accuracy: {evaluation['accuracy']:.2f}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error retraining model: {str(e)}")
            return False
    
    def export_transaction_history(self, file_path):
        """
        Export the transaction history to a CSV file
        
        Parameters:
        ----------
        file_path : str
            Path to save the transaction history
            
        Returns:
        -------
        bool
            True if export was successful
        """
        try:
            if self.transaction_history:
                history_df = pd.DataFrame(self.transaction_history)
                history_df.to_csv(file_path, index=False)
                self.logger.info(f"Exported {len(history_df)} transactions to {file_path}")
                return True
            else:
                self.logger.warning("No transaction history to export")
                return False
                
        except Exception as e:
            self.logger.error(f"Error exporting transaction history: {str(e)}")
            return False


# Example usage
def main():
    """
    Example of how to use the enhanced mapper
    """
    # Instantiate the mapper
    mapper = EnhancedKontoplanMapper()
    
    # Load the Standardkontoplan
    mapper.load_kontoplan('2023-01-31-Standardkontoplan.xlsx')
    
    # Try to load an existing model
    model_path = 'kontoplan_mapper_model.joblib'
    if os.path.exists(model_path):
        mapper.load_model(model_path)
    else:
        print("No existing model found. Please train a model first.")
        return
    
    # Load the answer file
    answer_file_path = 'transaction_mapping.csv'
    if os.path.exists(answer_file_path):
        mapper.load_answer_file(answer_file_path)
    else:
        print("No answer file found. Create one using the training_data_generator.py script.")
        return
    
    # Example of processing bank transactions
    new_transactions = pd.DataFrame({
        'date': pd.date_range(start='2023-02-01', periods=5),
        'description': [
            'BETALING SUPERMARKET A/S',
            'OVERFØRSEL FRA KUNDE',
            'HUSLEJE BETALING',
            'INDBETALING KUNDE Y',
            'LØNSYSTEM UDBETALING'
        ],
        'amount': [
            -1200.50,
            5000.00,
            -8500.00,
            12000.00,
            -25000.00
        ]
    })
    
    # Predict account categories
    predictions = mapper.predict_with_answer_file(new_transactions)
    
    print("\nPredictions for new transactions:")
    for i, row in predictions.iterrows():
        print(f"{row['description']} | Amount: {row['amount']} | " 
              f"Predicted account: {row['predicted_account']} | "
              f"Confidence: {row['confidence']:.2f} | "
              f"Source: {row['prediction_source']}")
    
    # Example of adding user feedback
    mapper.update_answer_file_with_feedback('HUSLEJE BETALING', 2210)  # Map to 'Lokaleomkostninger'
    
    # Save the updated answer file
    mapper.save_answer_file('transaction_mapping_updated.csv')
    
    # Check for uncertain predictions
    uncertain = mapper.get_uncertain_predictions()
    if uncertain:
        print("\nUncertain predictions that need review:")
        for pred in uncertain:
            print(f"{pred['description']} | Confidence: {pred['confidence']:.2f}")
    
    # Retrain the model if enough transaction history
    mapper.retrain_with_transaction_history(min_history_size=10)  # Lower threshold for example
    
    # Save the updated model
    mapper.save_model('kontoplan_mapper_model_updated.joblib')
    
    # Export transaction history
    mapper.export_transaction_history('transaction_history.csv')


if __name__ == "__main__":
    main()