import pandas as pd 
import numpy as np

def process_data (df : pd.DataFrame):

    # dropping unneccesary columns 
    columns_dropped = ['comments', 'Timestamp', 'state']
    df= df.drop(columns=columns_dropped , axis=1 , errors='ignore' )

    # handling missing values
    if 'self_employed' in df.columns:
        df['self_employed'].fillna(df['self_employed'].mode()[0], inplace=True)
        print ("self employed done")

    if 'work_interfere' in df.columns and 'treatment' in df.columns:
        df.loc[df['work_interfere'].isnull() & (df['treatment'] == 'No'), 
               'work_interfere'] = 'Not Applicable'
        df['work_interfere'] = df['work_interfere'].fillna(df['work_interfere'].mode()[0])
        print("work interfere done")

    male = ['male', 'man', 'm', 'mail', 'malr', 'cis male', 'cis man', 'male-ish','trans-female','Guy (-ish) ^_^',
             'something kinda male?','cis male','maile', 'mal', 'male (cis)', 'Make', 'male', 'man', 'msle', 'mail',
             'ostensibly male, unsure what that really means','neuter']
    female = ['female', 'woman', 'f', 'femake', 'cis female', 'cis woman','femail', 'female (cis)'
              ,'femake','female/woman']

    if 'Gender' in df.columns :
        df['Gender'] = df['Gender'].str.lower().str.strip()

        def clean_gender(g):
            if g in male:
                return 'Male'
            elif g in female:
                return 'Female'
            else:
                return 'Male'
        
        df['Gender'] = df['Gender'].apply(clean_gender)
        print ('gender done')
    
    if 'Age' in df.columns :
        filt = (df['Age']>=18) & (df['Age']<=70)
        df=df[filt]
        print(" age done")

    if 'no_employees' in df.columns :
        range_to_number = {
            '1-5': 3,             
            '6-25': 15,           
            '26-100': 63,         
            '100-500': 300,       
            '500-1000': 750,      
            'More than 1000': 1500
         }
        df['no_employees'] = df['no_employees'].map(range_to_number)
        df['no_employees'] = df['no_employees'].astype(int)
        print('employees done')

    # data augmenting 
    synthetic_df = df.sample(n=2000, replace=True, random_state=42)
    df = pd.concat([df, synthetic_df], ignore_index=True)

    return df

    





    


