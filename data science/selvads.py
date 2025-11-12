from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd 
import matplotlib.pyplot as plt
tips = pd.read_csv('student_exam_scores.csv') 
print(tips.head())
datasdt = tips[['previous_scores','exam_score']]
print(datasdt)
minmax = MinMaxScaler()
minmax = pd.DataFrame(minmax.fit_transform(datasdt),columns=['pre','scr'])
print(minmax)
dates = tips[['sleep_hours','attendance_percent']]
scalerstd = StandardScaler()
scalerstd = pd.DataFrame(scalerstd.fit_transform(dates),columns=['slp','atp'])
print(scalerstd)
plt.hist(minmax['scr'],bins=5,color='blue',edgecolor='black')
plt.xlabel('TOTAL PERCENTAGE')
plt.ylabel('TOTAL SCORE')
plt.title('STUDENT TOTAL MARKS')
plt.show()