import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class KontoplanMapper:
    """
    A machine learning-based mapper for categorizing bank transactions
    according to the Danish Standard Chart of Accounts (Standardkontoplan)
    """
    
    def __init__(self):
        self.model = None
        self.accounts_df = None
        self.account_categories = None
        self.label_encoder = None
        
    def load_kontoplan(self, excel_path):
        """
        Load the Danish Standardkontoplan from Excel file
        
        Parameters:
        ----------
        excel_path : str
            Path to the Excel file containing the chart of accounts
        """
        # Read the Kontoplan sheet
        try:
            kontoplan_df = pd.read_excel(excel_path, sheet_name="II) Kontoplan", header=None)
            print(f"Loaded kontoplan with {len(kontoplan_df)} entries")
            
            # Check if the structure is as expected
            if kontoplan_df.shape[1] < 3:
                raise ValueError("Kontoplan sheet does not have the expected structure")
            
            # Extract account information
            accounts = []
            for _, row in kontoplan_df.iterrows():
                # Skip rows without account numbers or descriptions
                if pd.notna(row[0]) and pd.notna(row[2]):
                    # If it's a header row, skip
                    if pd.notna(row[1]) and row[1] == 'Overskrift':
                        continue
                    
                    accounts.append({
                        'account_number': int(row[0]),
                        'description': str(row[2]).strip()
                    })
            
            self.accounts_df = pd.DataFrame(accounts)
            print(f"Extracted {len(self.accounts_df)} accounts from kontoplan")
            
            # Extract account categories
            categories = []
            current_main = ''
            for _, row in kontoplan_df.iterrows():
                if pd.notna(row[1]) and row[1] == 'Overskrift':
                    # Determine if it's a main or subcategory based on account number
                    acct_num = int(row[0])
                    if acct_num < 2000:  # Assuming main categories have lower numbers
                        current_main = str(row[2]).strip()
                        categories.append({
                            'level': 'main',
                            'name': current_main,
                            'account_prefix': acct_num // 1000 * 1000
                        })
                    else:
                        categories.append({
                            'level': 'sub',
                            'name': str(row[2]).strip(),
                            'main_category': current_main,
                            'account_prefix': acct_num // 100 * 100
                        })
            
            self.account_categories = pd.DataFrame(categories)
            print(f"Extracted {len(self.account_categories)} account categories")
            return True
        
        except Exception as e:
            print(f"Error loading kontoplan: {str(e)}")
            return False
    
    def preprocess_transactions(self, transactions_df):
        """
        Preprocess bank transactions for ML classification
        
        Parameters:
        ----------
        transactions_df : pandas.DataFrame
            DataFrame with bank transactions
            
        Returns:
        -------
        pandas.DataFrame
            Preprocessed transactions
        """
        df = transactions_df.copy()
        
        # Ensure we have required columns
        required_cols = ['description', 'amount']
        
        # Check if we have a 'description' column, if not try to find it
        if 'description' not in df.columns:
            # Look for columns with names like 'text', 'narrative', 'details', etc.
            desc_candidates = [col for col in df.columns if col.lower() in 
                              ['text', 'narrative', 'details', 'description', 'memo', 'note']]
            if desc_candidates:
                df['description'] = df[desc_candidates[0]]
            else:
                # If we can't find a description column, raise an error
                raise ValueError("No description column found in transaction data")
        
        # Check if we have an 'amount' column, if not try to find it
        if 'amount' not in df.columns:
            # Look for columns with names like 'amount', 'sum', 'total', etc.
            amount_candidates = [col for col in df.columns if col.lower() in 
                                ['amount', 'sum', 'total', 'beløb', 'værdi']]
            if amount_candidates:
                df['amount'] = df[amount_candidates[0]]
            else:
                # If we can't find an amount column, raise an error
                raise ValueError("No amount column found in transaction data")
        
        # Clean and normalize descriptions
        df['description'] = df['description'].astype(str)
        df['description'] = df['description'].apply(self._clean_text)
        
        # Extract features from the amount
        df['is_positive'] = df['amount'] > 0
        df['amount_abs'] = df['amount'].abs()
        
        # Add more features if available
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
        
        return df
    
    def _clean_text(self, text):
        """Clean and normalize transaction text"""
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and letters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_training_data(self, labeled_transactions):
        """
        Prepare training data from labeled transactions
        
        Parameters:
        ----------
        labeled_transactions : pandas.DataFrame
            DataFrame with transactions that have been manually labeled
            with account_number or account_category
            
        Returns:
        -------
        tuple
            X_train, X_test, y_train, y_test
        """
        # Preprocess transactions
        df = self.preprocess_transactions(labeled_transactions)
        
        # Check if we have labels
        if 'account_number' not in df.columns and 'account_category' not in df.columns:
            raise ValueError("No account labels found in the training data")
        
        # Use account_number as the target if available, otherwise use account_category
        target_col = 'account_number' if 'account_number' in df.columns else 'account_category'
        
        # Define features
        features = ['description']
        
        # Add amount-related features
        features.extend(['is_positive', 'amount_abs'])
        
        # Add date-related features if available
        if 'day_of_week' in df.columns:
            features.extend(['day_of_week', 'day_of_month', 'month'])
        
        # Split data
        X = df[features]
        y = df[target_col]
        
        # Save the label encoder mapping
        self.label_encoder = {label: idx for idx, label in enumerate(y.unique())}
        
        # Convert labels to numeric
        y_encoded = y.map(self.label_encoder)
        
        return train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    def train_model(self, X_train, y_train):
        """
        Train a machine learning model to classify transactions
        
        Parameters:
        ----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Training labels (encoded)
            
        Returns:
        -------
        bool
            True if training was successful
        """
        try:
            # Define the text processing pipeline
            text_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    min_df=2, 
                    max_df=0.95, 
                    ngram_range=(1, 2),
                    sublinear_tf=True
                ))
            ])
            
            # Process the description column
            text_features = text_pipeline.fit_transform(X_train['description'])
            
            # Get other features
            other_features = X_train.drop('description', axis=1).values
            
            # Combine text features with other features
            X_combined = np.hstack((text_features.toarray(), other_features))
            
            # Train a Random Forest classifier
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            clf.fit(X_combined, y_train)
            
            # Store the model and text pipeline
            self.model = {
                'text_pipeline': text_pipeline,
                'classifier': clf
            }
            
            print("Model training completed successfully")
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        
        Parameters:
        ----------
        X_test : pandas.DataFrame
            Test features
        y_test : pandas.Series
            Test labels (encoded)
            
        Returns:
        -------
        dict
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            # Process text features
            text_features = self.model['text_pipeline'].transform(X_test['description'])
            
            # Get other features
            other_features = X_test.drop('description', axis=1).values
            
            # Combine text features with other features
            X_combined = np.hstack((text_features.toarray(), other_features))
            
            # Make predictions
            y_pred = self.model['classifier'].predict(X_combined)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Reverse the label encoder to get original labels
            label_encoder_rev = {v: k for k, v in self.label_encoder.items()}
            y_test_original = [label_encoder_rev[label] for label in y_test]
            y_pred_original = [label_encoder_rev[label] for label in y_pred]
            
            return {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred),
                'y_test_original': y_test_original,
                'y_pred_original': y_pred_original
            }
            
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return None
    
    def predict(self, transactions):
        """
        Predict account categories for new transactions
        
        Parameters:
        ----------
        transactions : pandas.DataFrame
            DataFrame with bank transactions
            
        Returns:
        -------
        pandas.DataFrame
            Transactions with predicted account categories
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            # Preprocess transactions
            df = self.preprocess_transactions(transactions)
            
            # Process text features
            text_features = self.model['text_pipeline'].transform(df['description'])
            
            # Get other features
            other_features = df[['is_positive', 'amount_abs']].values
            
            # Add date-related features if available
            if 'day_of_week' in df.columns:
                other_features = np.hstack((
                    other_features, 
                    df[['day_of_week', 'day_of_month', 'month']].values
                ))
            
            # Combine text features with other features
            X_combined = np.hstack((text_features.toarray(), other_features))
            
            # Make predictions
            y_pred = self.model['classifier'].predict(X_combined)
            
            # Add predictions to the dataframe
            label_encoder_rev = {v: k for k, v in self.label_encoder.items()}
            df['predicted_account'] = [label_encoder_rev[label] for label in y_pred]
            
            # Add confidence scores
            y_proba = self.model['classifier'].predict_proba(X_combined)
            df['confidence'] = [max(proba) for proba in y_proba]
            
            # Add account descriptions if we have the kontoplan loaded
            if self.accounts_df is not None:
                # Create a mapping from account number to description
                account_map = dict(zip(
                    self.accounts_df['account_number'], 
                    self.accounts_df['description']
                ))
                
                # Add the description
                df['account_description'] = df['predicted_account'].map(account_map)
            
            return df
            
        except Exception as e:
            print(f"Error predicting accounts: {str(e)}")
            return None
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        try:
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'accounts_df': self.accounts_df,
                'account_categories': self.account_categories
            }
            
            joblib.dump(model_data, filepath)
            print(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.accounts_df = model_data['accounts_df']
            self.account_categories = model_data['account_categories']
            
            print(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False


# Example usage
def main():
    # Instantiate the mapper
    mapper = KontoplanMapper()
    
    # Load the Standardkontoplan
    mapper.load_kontoplan('2023-01-31-Standardkontoplan.xlsx')
    
    # Example of how to use with actual data
    # Step 1: Load labeled training data
    # This would be a CSV/Excel with transaction data and manually assigned account numbers
    # labeled_data = pd.read_csv('labeled_transactions.csv')
    
    # For demonstration, create a small sample of synthetic labeled data
    labeled_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'description': [
            'BETALING SUPERMARKET A/S',
            'OVERFØRSEL FRA KUNDE',
            'HUSLEJE BETALING',
            'INDBETALING KUNDE X',
            'LØNSYSTEM UDBETALING',
            'ERHVERVSFORSIKRING A/S',
            'MOBILTELEFON ABONNEMENT',
            'IT SERVICEAFTALE',
            'KØBT KONTORARTIKLER',
            'BETALING TRYKKERI'
        ] * 10,
        'amount': np.random.normal(0, 5000, 100),
        # Map to account numbers from the Standardkontoplan
        'account_number': np.random.choice([1010, 1390, 1810, 2010, 2210, 2830], 100)
    })
    
    # Step 2: Prepare the training data
    X_train, X_test, y_train, y_test = mapper.prepare_training_data(labeled_data)
    
    # Step 3: Train the model
    mapper.train_model(X_train, y_train)
    
    # Step 4: Evaluate the model
    evaluation = mapper.evaluate_model(X_test, y_test)
    print(f"Model accuracy: {evaluation['accuracy']:.2f}")
    
    # Step 5: Save the model for future use
    mapper.save_model('kontoplan_mapper_model.joblib')
    
    # Example of using the model to predict on new transactions
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
    
    predictions = mapper.predict(new_transactions)
    
    print("\nPredictions for new transactions:")
    for i, row in predictions.iterrows():
        print(f"{row['description']} | Amount: {row['amount']} | " 
              f"Predicted account: {row['predicted_account']} - "
              f"{row.get('account_description', 'N/A')} | "
              f"Confidence: {row['confidence']:.2f}")


if __name__ == "__main__":
    main()