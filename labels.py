import pandas as pd
import argparse
import re

# These are all present flairs from the subreddit
math_sci = ['Computer Science', 'Mathematics & Statistics', 'Software Engineering',
            'Computer Engineering', 'Electrical Engineering']
physical_sci = ['Atmospheric Science', 'Chemistry', 'Geology', 'Physics'
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
                           'Environment', 'Food Science', 'Human Nutrition',
                           'Plant Science', 'Geography', 'Urban Planning']
law = ['Law']

# TODO: keep going with the adding of different stems/acronyms
label_to_flairs = {'Math Sciences': math_sci + ['Mathematics', 'Math'],
                   'Physical Sciences': physical_sci,
                   'Life Sciences': life_sci,
                   'Management': management,
                   'Medicine': medicine + ['Med'],
                   'Music': music,
                   'Education': education,
                   'Humanities': humanities,
                   'Social Sciences': social_science + ['Poli Science', 'Soci', 'IDS', 'Econ'],
                   'Agriculture and Environment': agriculture_environment + ['Enviro'],
                   'Law': law}
for flairs in label_to_flairs.values():
    for flair in flairs:
        if 'Science' in flair:
            flairs.append(flair.replace('Science', 'Sci'))
        if 'Engineering' in flair:
            flairs.append(flair.replace('Engineering', 'Eng'))
        if 'Computer' in flair:
            flairs.append(flair.replace('Computer', 'Comp'))

strs_to_replace = ['honors', 'honours', 'joint']
regexes_to_replace = [r'U\d', r'\d{4}', r'\'\d{2}', r'B.?((Sc)|A|(Eng)|(Ed)|(Mus)|(Comm)).?']
def tokenize_flair_into_possible_majors(flair: str):
    flair = flair.lower()
    for str_to_repalce in strs_to_replace:
        flair = flair.replace(str_to_repalce, '')
    for regex_to_replace in regexes_to_replace:
        flair = re.sub(regex_to_replace, '', flair)
    tokens = [flair]
    # TODO: if there's a valid split on & or /, append all elements of the result to tokens
    # then, we will loop over and check all of the values of tokens against the values of label_to_flairs


def flair_to_label(row):
    flair = tokenize_flair_into_possible_majors(row['flair'])
    for label, possible_flairs in label_to_flairs.items():
        # TODO: for efficieny move the lower()ing outside of the outer loop somehow
        if flair.strip() in [possible_flair.lower() for possible_flair in possible_flairs]:
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