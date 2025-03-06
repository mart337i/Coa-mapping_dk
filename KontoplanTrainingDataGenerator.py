import pandas as pd
import numpy as np
import random
import os
import argparse
from datetime import datetime, timedelta
from faker import Faker
import re
import pandas as pd
import xlsxwriter
from pathlib import Path

class KontoplanTrainingDataGenerator:
    """
    Generate synthetic training data for mapping bank transactions 
    to the Danish Standardkontoplan
    """
    
    def __init__(self, kontoplan_path):
        """
        Initialize the generator with the path to the Standardkontoplan
        
        Parameters:
        ----------
        kontoplan_path : str
            Path to the Excel file containing the Danish Standardkontoplan
        """
        self.fake = Faker('da_DK')  # Use Danish locale
        self.kontoplan_path = kontoplan_path
        self.accounts = self._load_kontoplan()
        self.transaction_patterns = self._create_transaction_patterns()
        
    def _load_kontoplan(self):
        """
        Load and parse the Standardkontoplan from Excel file
        
        Returns:
        -------
        dict
            Dictionary mapping account numbers to their descriptions and categories
        """
        try:
            # Read the Kontoplan sheet
            kontoplan_df = pd.read_excel(self.kontoplan_path, sheet_name="II) Kontoplan", header=None)
            
            accounts = {}
            current_category = ""
            
            for idx, row in kontoplan_df.iterrows():
                # Skip rows with NaN values in key columns
                if pd.isna(row[0]):
                    continue
                
                account_number = int(row[0])
                
                # Check if it's a category header
                if pd.notna(row[1]) and row[1] == "Overskrift":
                    current_category = str(row[2]).strip()
                    continue
                
                # Only include actual account entries (not headers)
                if pd.notna(row[2]):
                    description = str(row[2]).strip()
                    
                    accounts[account_number] = {
                        'description': description,
                        'category': current_category
                    }
            
            print(f"Loaded {len(accounts)} accounts from Standardkontoplan")
            return accounts
        
        except Exception as e:
            print(f"Error loading kontoplan: {str(e)}")
            return {}

    def _create_transaction_patterns(self):
        """
        Create typical transaction description patterns for each account
        
        Returns:
        -------
        dict
            Dictionary mapping account numbers to lists of transaction patterns
        """
        patterns = {}
        
        # Patterns for sales-related accounts (1010-1590)
        sales_accounts = [acct for acct in self.accounts.keys() if 1000 <= acct < 2000]
        for acct in sales_accounts:
            desc = self.accounts[acct]['description'].lower()
            patterns[acct] = [
                f"faktura {{}} - {desc}",
                f"betaling fra {{}} a/s",
                f"indbetaling {{}}",
                f"overførsel fra {{}}",
                f"salg til {{}}",
                f"{{}} betaling",
                f"{{}} faktura nr. {{}}"
            ]
        
        # Patterns for raw materials and goods (1800-1890)
        if 1810 in self.accounts:
            patterns[1810] = [
                "købmand {}",
                "grossist {}",
                "materialer til {}",
                "råvarer {}",
                "indkøb af {}",
                "varekøb {}",
                "leverandør {}"
            ]
            
        # Patterns for sales expenses (2010-2090)
        if 2010 in self.accounts:
            patterns[2010] = [
                "rejseudgifter {}",
                "messe {}",
                "markedsføring {}",
                "annoncer {}",
                "reklame {}",
                "salgsfremmende {}",
                "repræsentation {}"
            ]
            
        # Patterns for premises expenses (2210-2290)
        if 2210 in self.accounts:
            patterns[2210] = [
                "husleje {}",
                "ejendomsskat {}",
                "el {}",
                "varme {}",
                "vand {}",
                "vedligeholdelse {}",
                "rengøring {}"
            ]
            
        # Patterns for administrative expenses (2830-2890)
        if 2830 in self.accounts:
            patterns[2830] = [
                "kontorartikler {}",
                "revisor {}",
                "advokat {}",
                "telefon {}",
                "internet {}",
                "software {}",
                "abonnement {}",
                "forsikring {}"
            ]
            
        # Patterns for payroll expenses (2990-3090)
        if 2990 in self.accounts:
            patterns[2990] = [
                "løn {}",
                "lønsystem {}",
                "lønudbetaling {}",
                "feriepenge {}",
                "pension {}",
                "personalegoder {}",
                "bonus {}"
            ]
        
        # Fill in patterns for accounts without specific patterns
        for acct in self.accounts:
            if acct not in patterns:
                desc = self.accounts[acct]['description'].lower()
                patterns[acct] = [
                    f"{desc} {{}}",
                    f"betaling {{}} - {desc}",
                    f"{{}} {desc}"
                ]
        return patterns
            
    def generate_transactions(self, num_transactions=1000, start_date=None, end_date=None):
        """
        Generate synthetic bank transactions with labels
        
        Parameters:
        ----------
        num_transactions : int
            Number of transactions to generate
        start_date : datetime, optional
            Start date for transactions (default: 1 year ago)
        end_date : datetime, optional
            End date for transactions (default: today)
            
        Returns:
        -------
        pandas.DataFrame
            DataFrame with synthetic transactions and their Kontoplan labels
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
            
        if end_date is None:
            end_date = datetime.now()
        
        # List of companies to use in transaction descriptions
        companies = [self.fake.company() for _ in range(50)]
        
        # Filter accounts that we'll use (exclude rarely used accounts)
        active_accounts = [acct for acct in self.accounts.keys() 
                          if len(self.transaction_patterns.get(acct, [])) > 0]
        
        # Generate transactions
        transactions = []
        
        for _ in range(num_transactions):
            # Pick a random account
            account_number = random.choice(active_accounts)
            account_info = self.accounts[account_number]
            
            # Determine if it's income or expense based on account type
            is_income = 1000 <= account_number < 1600
            
            # Generate amount (negative for expenses, positive for income)
            base_amount = random.uniform(100, 50000)
            amount = base_amount if is_income else -base_amount
            
            # Generate date
            days_range = (end_date - start_date).days
            random_days = random.randint(0, max(0, days_range))
            date = start_date + timedelta(days=random_days)
            
            description = ""
            # Generate description using patterns
            patterns = self.transaction_patterns.get(account_number, [])
            if patterns:
                pattern = random.choice(patterns)
                company = random.choice(companies)
                
                # Sometimes add an invoice or reference number
                if random.random() < 0.3:
                    description = f" {pattern} - ref. {random.randint(10000, 99999)}"
            else:
                # Fallback if no patterns
                description = f"{account_info['description']} - {random.choice(companies)}"
            
            # Create transaction record
            transaction = {
                'date': date,
                'description': description,
                'amount': round(amount, 2),
                'account_number': account_number,
                'account_description': account_info['description'],
                'account_category': account_info['category']
            }
            
            transactions.append(transaction)
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Sort by date
        df = df.sort_values('date')
        
        return df
    
    def save_training_data(self, df, output_path):
        """
        Save generated training data to CSV and Excel files
        
        Parameters:
        ----------
        df : pandas.DataFrame
            Generated transaction data
        output_path : str
            Base path for output files (without extension)
        """
        # Save as CSV
        csv_path = f"{output_path}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Training data saved to {csv_path}")
        
        # Save as Excel
        excel_path = f"{output_path}.xlsx"
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Transactions', index=False)
            
            # Add a sheet with account reference
            accounts_df = pd.DataFrame([
                {'account_number': acct, 
                 'description': self.accounts[acct]['description'],
                 'category': self.accounts[acct]['category']}
                for acct in sorted(self.accounts.keys())
            ])
            accounts_df.to_excel(writer, sheet_name='Accounts', index=False)
            
        print(f"Training data saved to {excel_path}")
        
    def create_answer_file(self, df, output_path):
        """
        Create an answer file with correct mappings for training
        
        Parameters:
        ----------
        df : pandas.DataFrame
            Generated transaction data
        output_path : str
            Path for the answer file
        """
        # Create a mapping file that can be used to train the model
        mapping_data = []
        
        # Get unique transaction patterns
        unique_patterns = df[['description', 'account_number']].drop_duplicates()
        
        for _, row in unique_patterns.iterrows():
            # Get the most common words in the description
            words = re.findall(r'\w+', row['description'].lower())
            keywords = ' '.join(sorted(set(words), key=words.count, reverse=True)[:5])
            
            mapping_data.append({
                'pattern': row['description'],
                'keywords': keywords,
                'account_number': row['account_number'],
                'account_description': self.accounts[row['account_number']]['description']
            })
        
        # Create DataFrame and save
        mapping_df = pd.DataFrame(mapping_data)
        mapping_df.to_csv(output_path, index=False)
        print(f"Answer file created at {output_path}")

def main():
    """Main function to run the training data generator"""
    parser = argparse.ArgumentParser(description='Generate training data for Kontoplan mapping')
    parser.add_argument('--kontoplan', required=True, help='Path to Standardkontoplan Excel file')
    parser.add_argument('--num_transactions', type=int, default=1000, help='Number of transactions to generate')
    parser.add_argument('--output', default='training_data', help='Base output filename (without extension)')
    parser.add_argument('--answer_file', default='transaction_mapping.csv', help='Filename for answer file')
    
    args = parser.parse_args()
    
    # Create the generator
    generator = KontoplanTrainingDataGenerator(args.kontoplan)
    
    # Generate transactions
    transactions = generator.generate_transactions(args.num_transactions)
    
    # Save the data
    generator.save_training_data(transactions, args.output)
    
    # Create answer file
    generator.create_answer_file(transactions, args.answer_file)

if __name__ == "__main__":
    main()