
import pandas as pd


path1 = '/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/data.csv'
path2 = '/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/heart_cleveland_upload.csv'
path3 = '/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/heart_ED.csv'
path4 = '/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/statlog.csv'
path5 = '/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/heart_statlog_cleveland_hungary_final.csv'

file1 = pd.read_csv(path1)
file2 = pd.read_csv(path2)
file3 = pd.read_csv(path3)
file4 = pd.read_csv(path4)
file5 = pd.read_csv(path5)


print(file1.shape, "\n")
print(file2.shape, "\n")
print(file3.shape, "\n")
print(file4.shape, "\n")
print(file5.shape, "\n")


print(file1.columns, "\n")
print(file2.columns, "\n")
print(file3.columns, "\n")
print(file4.columns, "\n")
print(file5.columns, "\n")


file1.drop(columns=['ca', 'thal'], inplace=True)
file2.drop(columns=['ca', 'thal'], inplace=True)
file4.drop(columns=['ca', 'thal'], inplace=True)


file1.rename(columns={"num       ":"target"}, inplace=True)
file2.rename(columns={"condition":"target"}, inplace=True)
file3.rename(columns={'Age':'age',
                      'Sex':'sex',
                      'ChestPainType':'cp',
                      'RestingBP':'trestbps',
                      'Cholesterol':'chol',
                      'FastingBS':'fbs',
                      'RestingECG':'restecg',
                      'MAX HR':'thalach', 
                      'ExerciseAngina':'exang',
                      'Oldpeak':'oldpeak',
                      'ST_Slope':'slope',
                      'HeartDisease':'target'}, inplace=True)
file4.rename(columns={"presence":"target"}, inplace=True)
file5.rename(columns={
                      'chest pain type':'cp',
                      'resting bp s':'trestbps',
                      'cholesterol':'chol',
                      'fasting blood sugar':'fbs',
                      'resting ecg':'restecg',
                      'max heart rate':'thalach', 
                      'exercise angina':'exang',
                      'ST slope':'slope'}, inplace=True)


file1['target'].value_counts()
file2['target'].value_counts()
file3['target'].value_counts()
file4['target'].value_counts()
file5['target'].value_counts()


file4.replace({'target' : { 1 : 0, 
                            2 : 1 }}, inplace=True)

df = pd.concat([file1, file2, file3, file4, file5], axis=0, ignore_index=True)
df.shape

df.columns


dup_rows = df[df.duplicated()]
print("No. of duplicate rows: ", dup_rows.shape[0])


data = df.sample(frac=1, random_state=44)


data.to_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/merge_data.csv', index=False)

