import pandas as pd
import argparse
import nltk
from nltk.corpus import stopwords
import string
import re

# These are all present flairs from the subreddit
math_sci = ['Computer Science', 'Mathematics & Statistics', 'Software Engineering',
            'Computer Engineering', 'Electrical Engineering']
physical_sci = ['Atmospheric Science', 'Chemistry', 'Geology', 'Physics',
                'Chemical Engineering', 'Civil Engineering', 'Materials Engineering',
                'Mechanical Engineering', 'Mining Engineering', 'Architecture']
life_sci = ['Anatomy & Cell Biology', 'Biochemistry', 'Biology', 'Physiology',          
            'Kinesiology', 'Microbiology & Immunology', 'Neuroscience', 'Bioengineering',
            'Psychology']
management = ['Management', 'Accounting', 'Business Administration', 'Finance'
              'Marketing', 'Organizational Behaviour', 'Strategic Management']
medicine = ['Pharmacology', 'Medicine', 'Dentistry', 'Nursing']
music = ['Music']
education = ['Physical & Occupational Therapy', 'Education', 'Primary Education',        
             'Secondary Education']
humanities = ['East Asian Studies', 'English', 'Français', 'Gender Studies',
              'History & Classics', 'Islamic Studies', 'Jewish Studies', 'Philosophy',
              'Religious Studies', 'Art History & Communications', 'Linguistics']
social_science = ['Anthropology', 'Economics', 'Industrial Relations',
                  'International Development', 'Political Science', 'Sociology',
                  'Urban Systems']
agriculture_environment = ['Environment', 'Agricultural & Environmental Science',
                           'Animal Science', 'Bioresource Engineering', 'Ecology',
                           'Food Science', 'Human Nutrition',
                           'Plant Science', 'Geography', 'Urban Planning']
law = ['Law']

# TODO: keep going with the adding of different stems/acronyms
label_to_flairs = {'Math Sciences': math_sci + ['Mathematics', 'Math', 'CS'],
                   'Physical Sciences': physical_sci,
                   'Life Sciences': life_sci + ['Psych'],
                   'Management': management,
                   'Medicine': medicine + ['Med'],
                   'Music': music,
                   'Education': education,
                   'Humanities': humanities,
                   'Social Sciences': social_science + ['Poli Science', 'Poli', 'Soci', 'IDS', 'Econ'],
                   'Agriculture and Environment': agriculture_environment + ['Enviro', 'Environmental Science'],
                   'Law': law}
for label in label_to_flairs:
    label_to_flairs[label] = [flair.lower() for flair in label_to_flairs[label]]
    for flair in label_to_flairs[label]:
        if 'science' in flair:
            label_to_flairs[label].append(flair.replace('science', 'sci'))
        if 'engineering' in flair:
            label_to_flairs[label].append(flair.replace('engineering', 'eng'))
        if 'computer' in flair:
            label_to_flairs[label].append(flair.replace('computer', 'comp'))
        if 'biology' in flair:
            label_to_flairs[label].append(flair.replace('biology', 'bio'))
        if 'chemistry' in flair:
            label_to_flairs[label].append(flair.replace('chemistry', 'chem'))

def tokenize_flair(flair: str):
    flair = flair.lower()
    all_ngrams = [flair]
    cleaned_flair = flair
    for punc in string.punctuation:
        cleaned_flair = cleaned_flair.replace(punc, '')
    cleaned_flair = ' '.join((token for token in cleaned_flair.split()
                            if token not in set(stopwords.words('english'))))
    for flair in [flair, cleaned_flair]:
        split_flair = flair.split()
        for n in range(1, len(split_flair) + 1):
            ngrams = nltk.ngrams(split_flair, n)
            for ngram in ngrams:
                all_ngrams.append(' '.join(ngram))
    return all_ngrams

def flair_to_label(row):
    flair_tokens = tokenize_flair(row['flair'])
    for label, possible_flairs in label_to_flairs.items():
        # print(flair_tokens, possible_flairs)
        for flair_token in flair_tokens:
            if flair_token in possible_flairs:
                return label
    return 'UNKNOWN'

def comment_flairs_to_label(in_csvs, out_csv):
    dfs = []
    for comment_csv in in_csvs:
        dfs.append(pd.read_csv(comment_csv))
    df = pd.concat(dfs)
    df['label'] = df.apply(flair_to_label, axis=1)
    print('Number of each label:')
    print(df['label'].value_counts())
    df.to_csv(out_csv, index=False)


# Some notes that may be interesting for the report: from the two original datasets, naive w/ just the preset flairs was 4956 unknown, with all the above rules it's down to 4269
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--comments_csvs', 
        nargs='+',
        type=str,
        required=True
    )
    parser.add_argument(
        '--flairs_csv', 
        type=str,
        default="data/dataset.csv"
    )
    args = parser.parse_args()

    comment_flairs_to_label(args.comments_csvs, args.flairs_csv)