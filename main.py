import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class DanishKontoplanMapper:
    def __init__(self, kontoplan_path=None):
        """
        Initialize the Kontoplan Mapper with a Danish chart of accounts.
        
        Args:
            kontoplan_path (str): Path to JSON file containing the Danish kontoplan structure
        """
        self.model = None
        self.vectorizer = None
        self.kontoplan = self._load_kontoplan(kontoplan_path)
        self.mapping_rules = self._define_mapping_rules()
        
    def _load_kontoplan(self, kontoplan_path):
        """
        Load the Danish kontoplan from a JSON file or use a default simplified version.
        """
        if kontoplan_path and os.path.exists(kontoplan_path):
            with open(kontoplan_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Simplified default Danish kontoplan structure
            return {
                "1000-1999": "Aktiver (Assets)",
                "1000-1099": "Immaterielle anlægsaktiver (Intangible fixed assets)",
                "1100-1299": "Materielle anlægsaktiver (Tangible fixed assets)",
                "1300-1399": "Finansielle anlægsaktiver (Financial fixed assets)",
                "1400-1699": "Varebeholdninger (Inventory)",
                "1700-1799": "Tilgodehavender fra salg (Trade receivables)",
                "1800-1899": "Andre tilgodehavender (Other receivables)",
                "1900-1999": "Likvide beholdninger (Cash and cash equivalents)",
                "2000-2999": "Passiver (Liabilities and Equity)",
                "2000-2099": "Egenkapital (Equity)",
                "2100-2199": "Hensættelser (Provisions)",
                "2200-2399": "Langfristet gæld (Long-term debt)",
                "2400-2699": "Kortfristet gæld (Short-term debt)",
                "2700-2799": "Moms og afgifter (VAT and duties)",
                "2800-2999": "Anden gæld (Other payables)",
                "3000-3999": "Omsætning (Revenue)",
                "4000-4999": "Direkte omkostninger (Direct costs)",
                "5000-5999": "Eksterne omkostninger (External expenses)",
                "6000-6999": "Personaleomkostninger (Personnel expenses)",
                "7000-7999": "Af- og nedskrivninger (Depreciation and amortization)",
                "8000-8999": "Finansielle poster (Financial items)",
                "9000-9999": "Ekstraordinære poster og skat (Extraordinary items and tax)"
            }
    
    def _define_mapping_rules(self):
        """
        Define rules for mapping transactions to kontoplan categories.
        These are simplified examples and would need to be expanded for real use.
        """
        return {
            # Revenue
            "3000": ["indbetalinger", "salg", "faktura", "betaling modtaget", "honorar"],
            
            # Direct costs
            "4000": ["varekøb", "råvarer", "direkte materialer"],
            
            # External expenses
            "5000": ["husleje", "kontor", "el", "vand", "varme"],
            "5100": ["telefon", "internet", "mobil"],
            "5200": ["revisor", "advokat", "konsulent"],
            "5300": ["rejse", "hotel", "transport", "billetter"],
            "5400": ["marketing", "reklame", "annonce"],
            
            # Personnel expenses
            "6000": ["løn", "gager", "honorar", "feriepenge"],
            "6100": ["pension", "arbejdsgiver", "arbejdsmarkedsbidrag"],
            
            # Financial items
            "8000": ["renter", "gebyr", "bank", "udbytte"],
            
            # VAT and duties
            "2700": ["moms", "skat", "afgift"]
        }
    
    def preprocess_transactions(self, df):
        """
        Preprocess bank transaction data.
        
        Args:
            df (pd.DataFrame): DataFrame with bank transaction data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Ensure we have the necessary columns
        required_cols = ['date', 'amount', 'description']
        for col in required_cols:
            if col not in processed_df.columns:
                raise ValueError(f"Required column '{col}' not found in transaction data")
        
        # Convert date to datetime if it isn't already
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        
        # Normalize description text
        processed_df['description_normalized'] = processed_df['description'].str.lower()
        
        # Add feature for transaction type (positive = income, negative = expense)
        processed_df['transaction_type'] = processed_df['amount'].apply(
            lambda x: 'income' if x > 0 else 'expense'
        )
        
        return processed_df
    
    def rule_based_mapping(self, transactions_df):
        """
        Apply rule-based mapping to assign kontoplan accounts to transactions.
        
        Args:
            transactions_df (pd.DataFrame): Preprocessed transaction data
            
        Returns:
            pd.DataFrame: Transactions with added kontoplan_account column
        """
        df = transactions_df.copy()
        df['kontoplan_account'] = None
        df['kontoplan_category'] = None
        
        # For each transaction, check if its description matches any of our rules
        for idx, row in df.iterrows():
            description = row['description_normalized']
            
            # Check each kontoplan account's keywords
            for account, keywords in self.mapping_rules.items():
                if any(keyword in description for keyword in keywords):
                    df.at[idx, 'kontoplan_account'] = account
                    
                    # Find the category for this account
                    for category_range, category_name in self.kontoplan.items():
                        start, end = map(int, category_range.split('-'))
                        if start <= int(account) <= end and '-' in category_range:
                            df.at[idx, 'kontoplan_category'] = category_name
                            break
                    
                    break
        
        return df
    
    def train_ml_classifier(self, labeled_data):
        """
        Train a machine learning model to predict kontoplan accounts based on transaction descriptions.
        
        Args:
            labeled_data (pd.DataFrame): Transaction data with known kontoplan_account labels
            
        Returns:
            bool: True if training was successful
        """
        # Drop rows where kontoplan_account is None
        labeled_data = labeled_data.dropna(subset=['kontoplan_account'])
        
        if len(labeled_data) < 10:  # Need sufficient data to train
            print("Not enough labeled data to train the model")
            return False
        
        # Prepare features and target
        X = labeled_data['description_normalized']
        y = labeled_data['kontoplan_account']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(max_features=1000)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train a Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_tfidf)
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
        
        return True
    
    def predict_accounts(self, transactions_df):
        """
        Use trained ML model to predict kontoplan accounts for transactions.
        
        Args:
            transactions_df (pd.DataFrame): Transaction data
            
        Returns:
            pd.DataFrame: Transactions with predicted kontoplan accounts
        """
        df = transactions_df.copy()
        
        if self.model is None or self.vectorizer is None:
            print("Model not trained yet. Using rule-based mapping only.")
            return self.rule_based_mapping(df)
        
        # Apply rule-based mapping first
        df = self.rule_based_mapping(df)
        
        # For transactions without a rule-based mapping, use ML predictions
        mask = df['kontoplan_account'].isna()
        if mask.any():
            # Get the descriptions needing prediction
            descriptions = df.loc[mask, 'description_normalized']
            
            # Transform text to TF-IDF features
            X_tfidf = self.vectorizer.transform(descriptions)
            
            # Make predictions
            predictions = self.model.predict(X_tfidf)
            
            # Update the DataFrame with predictions
            df.loc[mask, 'kontoplan_account'] = predictions
            
            # Also update the category based on the predicted account
            for idx, account in enumerate(predictions):
                orig_idx = df.loc[mask].iloc[idx].name
                for category_range, category_name in self.kontoplan.items():
                    start, end = map(int, category_range.split('-'))
                    if start <= int(account) <= end and '-' in category_range:
                        df.at[orig_idx, 'kontoplan_category'] = category_name
                        break
        
        return df
    
    def generate_report(self, mapped_transactions):
        """
        Generate a report based on kontoplan-mapped transactions.
        
        Args:
            mapped_transactions (pd.DataFrame): Transactions with kontoplan mapping
            
        Returns:
            dict: Report data
        """
        report = {}
        
        # Ensure we have the kontoplan mapping
        if 'kontoplan_account' not in mapped_transactions.columns:
            raise ValueError("Transactions must be mapped to kontoplan accounts first")
        
        # Group transactions by kontoplan category and account
        category_summary = mapped_transactions.groupby('kontoplan_category')['amount'].sum()
        account_summary = mapped_transactions.groupby('kontoplan_account')['amount'].sum()
        
        # Monthly trend analysis
        mapped_transactions['month'] = mapped_transactions['date'].dt.to_period('M')
        monthly_trend = mapped_transactions.groupby(['month', 'kontoplan_category'])['amount'].sum().unstack()
        
        report['category_summary'] = category_summary.to_dict()
        report['account_summary'] = account_summary.to_dict()
        report['monthly_trend'] = monthly_trend.to_dict()
        
        return report
    
    def visualize_data(self, mapped_transactions, output_path=None):
        """
        Create visualizations for the mapped transaction data.
        
        Args:
            mapped_transactions (pd.DataFrame): Transactions with kontoplan mapping
            output_path (str): Directory to save visualizations
            
        Returns:
            dict: Paths to generated visualizations
        """
        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)
        
        viz_paths = {}
        
        # 1. Category distribution pie chart
        plt.figure(figsize=(12, 8))
        category_totals = mapped_transactions.groupby('kontoplan_category')['amount'].sum().abs()
        plt.pie(category_totals, labels=category_totals.index, autopct='%1.1f%%')
        plt.title('Transaction Distribution by Kontoplan Category')
        if output_path:
            path = os.path.join(output_path, 'category_distribution.png')
            plt.savefig(path)
            viz_paths['category_distribution'] = path
        plt.close()
        
        # 2. Monthly trends by category
        plt.figure(figsize=(15, 10))
        mapped_transactions['month'] = mapped_transactions['date'].dt.to_period('M')
        monthly_data = mapped_transactions.groupby(['month', 'kontoplan_category'])['amount'].sum().unstack()
        monthly_data.plot(kind='bar', stacked=True)
        plt.title('Monthly Transaction Totals by Category')
        plt.xlabel('Month')
        plt.ylabel('Amount (DKK)')
        if output_path:
            path = os.path.join(output_path, 'monthly_trends.png')
            plt.savefig(path)
            viz_paths['monthly_trends'] = path
        plt.close()
        
        # 3. Income vs. Expenses
        plt.figure(figsize=(10, 6))
        income_expense = mapped_transactions.copy()
        income_expense['type'] = income_expense['amount'].apply(lambda x: 'Income' if x > 0 else 'Expense')
        type_monthly = income_expense.groupby(['month', 'type'])['amount'].sum().abs().unstack()
        type_monthly.plot(kind='line', marker='o')
        plt.title('Income vs. Expenses by Month')
        plt.xlabel('Month')
        plt.ylabel('Amount (DKK)')
        if output_path:
            path = os.path.join(output_path, 'income_vs_expenses.png')
            plt.savefig(path)
            viz_paths['income_vs_expenses'] = path
        plt.close()
        
        return viz_paths
    
    def export_to_excel(self, mapped_transactions, output_path):
        """
        Export the mapped transactions to Excel in a format suitable for accounting.
        
        Args:
            mapped_transactions (pd.DataFrame): Transactions with kontoplan mapping
            output_path (str): Path to save the Excel file
            
        Returns:
            str: Path to the exported Excel file
        """
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Sheet 1: All transactions with mapping
            mapped_transactions.to_excel(writer, sheet_name='All Transactions', index=False)
            
            # Sheet 2: Summary by Kontoplan category
            category_summary = mapped_transactions.pivot_table(
                values='amount', 
                index='kontoplan_category',
                aggfunc='sum'
            )
            category_summary.to_excel(writer, sheet_name='Category Summary')
            
            # Sheet 3: Summary by Kontoplan account
            account_summary = mapped_transactions.pivot_table(
                values='amount', 
                index=['kontoplan_account', 'kontoplan_category'],
                aggfunc='sum'
            )
            account_summary.to_excel(writer, sheet_name='Account Summary')
            
            # Sheet 4: Monthly summary
            mapped_transactions['month'] = mapped_transactions['date'].dt.to_period('M')
            monthly_summary = mapped_transactions.pivot_table(
                values='amount',
                index='month',
                columns='kontoplan_category',
                aggfunc='sum',
                fill_value=0
            )
            monthly_summary.to_excel(writer, sheet_name='Monthly Summary')
        
        return output_path


# Example usage
if __name__ == "__main__":
    # Sample transaction data (in a real scenario, this would come from an API)
    sample_data = {
        'date': [
            '2023-01-15', '2023-01-20', '2023-01-25', '2023-02-01', 
            '2023-02-10', '2023-02-15', '2023-02-20', '2023-03-01'
        ],
        'amount': [
            -5000, 15000, -1200, -3500, 
            20000, -800, -2500, -4000
        ],
        'description': [
            'Husleje januar', 'Faktura #1234 betaling', 'El-regning', 'Indkøb kontormøbler',
            'Betaling for konsulentarbejde', 'Telefon og internet', 'Materialer til projekt', 'Løn til medarbejder'
        ]
    }
    
    # Create DataFrame
    transactions_df = pd.DataFrame(sample_data)
    
    # Initialize mapper
    mapper = DanishKontoplanMapper()
    
    # Preprocess transactions
    processed_df = mapper.preprocess_transactions(transactions_df)
    
    # Apply rule-based mapping
    mapped_df = mapper.rule_based_mapping(processed_df)
    
    # Check which transactions were mapped
    print("Mapped transactions:")
    print(mapped_df[['description', 'kontoplan_account', 'kontoplan_category']])
    
    # In a real scenario, you would have labeled data to train the model
    # Here we'll simulate that by using the rule-based mapping as training data
    mapper.train_ml_classifier(mapped_df.dropna(subset=['kontoplan_account']))
    
    # Now use the ML model to predict accounts for all transactions
    final_mapped_df = mapper.predict_accounts(processed_df)
    
    # Generate report
    report = mapper.generate_report(final_mapped_df)
    print("\nCategory summary:")
    for category, amount in report['category_summary'].items():
        if category:  # Skip None values
            print(f"{category}: {amount} DKK")
    
    # Visualize the data
    viz_paths = mapper.visualize_data(final_mapped_df, 'output')
    print("\nVisualizations created at:", viz_paths)
    
    # Export to Excel
    excel_path = mapper.export_to_excel(final_mapped_df, 'danish_kontoplan_mapping.xlsx')
    print("\nExcel report exported to:", excel_path)
