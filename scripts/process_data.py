import os
from pathlib import Path

import pandas as pd

PATH = Path(__file__).resolve().parent.parent

col_names = [
    'checking_account_status',
    'duration_months',
    'credit_history',
    'purpose',
    'credit_amount',
    'savings_account_bonds',
    'present_employment_since',
    'installment_rate_pct_of_disp_income',
    'personal_status_sex',
    'other_debtors_guarantors',
    'present_residence_since',
    'property',
    'age_years',
    'other_installment_plans',
    'housing',
    'existing_credits_count',
    'job',
    'people_liable_for_maintenance',
    'telephone',
    'foreign_worker',
    'class'
]

def get_mappings():
    # Mappings based on data/raw/description.txt
    return {
        'checking_account_status': {
            'A11': '< 0 DM',
            'A12': '0 <= ... < 200 DM',
            'A13': '>= 200 DM / salary assign.',
            'A14': 'no checking account',
        },
        'credit_history': {
            'A30': 'no credits taken/all credits paid back duly',
            'A31': 'all credits at this bank paid back duly',
            'A32': 'existing credits paid duly',
            'A33': 'delay in paying off in past',
            'A34': 'critical/other credits exist',
        },
        'purpose': {
            'A40': 'car (new)',
            'A41': 'car (used)',
            'A42': 'furniture/equipment',
            'A43': 'radio/television',
            'A44': 'domestic appliances',
            'A45': 'repairs',
            'A46': 'education',
            'A47': 'vacation',
            'A48': 'retraining',
            'A49': 'business',
            'A410': 'others',
        },
        'savings_account_bonds': {
            'A61': '< 100 DM',
            'A62': '100 - 500 DM',
            'A63': '500 - 1000 DM',
            'A64': '>= 1000 DM',
            'A65': 'unknown/no savings account',
        },
        'present_employment_since': {
            'A71': 'unemployed',
            'A72': '< 1 year',
            'A73': '1 <= ... < 4 years',
            'A74': '4 <= ... < 7 years',
            'A75': '>= 7 years',
        },
        'personal_status_sex': {
            'A91': 'male: divorced/separated',
            'A92': 'female: div/sep/married',
            'A93': 'male: single',
            'A94': 'male: married/widowed',
            'A95': 'female: single',
        },
        'other_debtors_guarantors': {
            'A101': 'none',
            'A102': 'co-applicant',
            'A103': 'guarantor',
        },
        'property': {
            'A121': 'real estate',
            'A122': 'bldg society/life ins.',
            'A123': 'car or other',
            'A124': 'unknown/no property',
        },
        'other_installment_plans': {
            'A141': 'bank',
            'A142': 'stores',
            'A143': 'none',
        },
        'housing': {
            'A151': 'rent',
            'A152': 'own',
            'A153': 'for free',
        },
        'job': {
            'A171': 'unemployed/unskilled non-res.',
            'A172': 'unskilled resident',
            'A173': 'skilled employee/official',
            'A174': 'management/self-employed/etc',
        },
        'telephone': {
            'A191': 'none',
            'A192': 'yes, registered',
        },
        'foreign_worker': {
            'A201': 'yes',
            'A202': 'no',
        },
        'class': {
            '1': 1,
            '2': 0,
        },
    }


def create_german_credit_csv():
    """Read german.data and create german_credit.csv with mappings."""
    raw_path = PATH / 'data' / 'raw' / 'german.data'
    out_dir = PATH / 'data' / 'processed'
    out_path = out_dir / 'german_credit.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Read raw data
    rows = []
    with open(raw_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                rows.append(parts)
    
    if not rows:
        raise RuntimeError('No data found in german.data')

    if len(rows[0]) != len(col_names):
        raise ValueError(
            f'Expected {len(col_names)} columns, got {len(rows[0])}'
        )

    # Create DataFrame with your exact column order
    df = pd.DataFrame(rows, columns=col_names)

    # Convert numeric columns to appropriate types
    numeric_cols = [
        'duration_months', 'credit_amount',
        'installment_rate_pct_of_disp_income',
        'present_residence_since', 'age_years', 'existing_credits_count',
        'people_liable_for_maintenance'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='raise')

    # Apply mappings to categorical columns
    mappings = get_mappings()
    categorical_cols = [col for col in col_names if col not in numeric_cols]

    for col in categorical_cols:
        if col in df.columns and col in mappings:
            df[col] = df[col].map(mappings[col]).fillna(df[col])

    # Save to CSV
    df.to_csv(out_path, index=False)
    print(f'Created {out_path} with shape {df.shape}')
    return df


if __name__ == '__main__':
    create_german_credit_csv()
