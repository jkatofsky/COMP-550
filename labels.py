import pandas as pd
import argparse

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

# TODO: programmatically add variations here; science -> sci, engineering -> eng
label_to_flairs = {'Math Sciences': math_sci,
                   'Physical Sciences': physical_sci,
                   'Life Sciences': life_sci,
                   'Management': management,
                   'Medicine': medicine,
                   'Music': music,
                   'Education': education,
                   'Humanities': humanities,
                   'Social Sciences': social_science,
                   'Agriculture and Environment': agriculture_environment,
                   'Law': law}

# TODO: figure out a way to add stems/acryonyms such as CS, poli, econ, IDS, etc. May require tokenization

def flair_to_label(row):
    flair = row['flair']
    for label, possible_flairs in label_to_flairs.items():
        if any(possible_flair.lower() in flair.lower() for possible_flair in possible_flairs):
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