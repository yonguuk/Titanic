import csv as csv
import numpy as np

#csv_file_object = csv.reader(open('C:/Users/yonguk/Desktop/kaggle/train.csv','rb'))
file_path='C:/Users/yonguk/Desktop/kaggle/'
csv_file_object=csv.reader(open('C:/Users/yonguk/Desktop/kaggle/train.csv','rU'))
header=csv_file_object.__next__()
data=[]
for row in csv_file_object:
    data.append(row)
data=np.array(data)
print(data[-1])
number_passangers = np.size(data[0:,1].astype(np.float))
print("number passangers = " + str(number_passangers))
number_survived=np.sum(data[0:,1].astype(np.float))
proportion_suvivors=number_survived/number_passangers
print('proportion survived : ' + str(proportion_suvivors))

women_only_stats = data[0:,4] == 'female' #find only 'female'
men_only_stats = data[0:,4] == "male"

women_onboard = data[women_only_stats,1].astype(np.float)
print(women_onboard)
men_onboard = data[men_only_stats,1].astype(np.float)

proportion_women_survived=np.sum(women_onboard)/np.size(women_onboard)
print('proportion women survived : ' + str(proportion_women_survived))
proportion_men_survived = np.sum(men_onboard)/np.size(men_onboard)
#print('proportion men survived : ' + str(proportion_men_survived))
print('proportion men survived : %s' % proportion_men_survived)

#gender based model
"""
test_file=open(file_path+'test.csv','rU')
test_file_object = csv.reader(test_file)
header=test_file_object.__next__();

#prediction_file=open('gender_based_model.csv','wb')
prediction_file=open('gender_based_model.csv','w',newline="")
prediction_file_objet = csv.writer(prediction_file)

prediction_file_objet.writerow(['PassengerId','Survived'])
for row in test_file_object:
    if row[3] == 'female':
        prediction_file_objet.writerow([row[0],'1'])
    else:
        prediction_file_objet.writerow([row[0],'0'])
test_file.close()
prediction_file.close()
"""

fare_ceiling=40
data[data[0:,9].astype(np.float) >= fare_ceiling,9] = fare_ceiling-1.0
print(data[0:,9])

fare_bracket_size = 10
number_of_price_brackets=fare_ceiling/fare_bracket_size
print(number_of_price_brackets)
number_of_classes = 3

#second submission
"""
fare_ceiling = 40
data[data[0::,9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling-1.0
fare_bracket_size = 10
number_of_price_brackets=fare_ceiling//fare_bracket_size
number_of_classes=3
number_of_classes = len(np.unique(data[0::,2])) #calculate this from the data directly

survival_table=np.zeros((2, number_of_classes, number_of_price_brackets))



#xrange has been removed in python 3.5
for i in range(number_of_classes):
    for j in range(number_of_price_brackets):

        women_only_stats = data[
            (data[0::,4] == 'female')
            & (data[0::,2].astype(np.float)==i+1)
            & (data[0:,9].astype(np.float)>=j*fare_bracket_size)
            & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size),1]

        men_only_stats = data[
            (data[0::,4] == 'male')
            & (data[0::,2].astype(np.float)==i+1)
            & (data[0:,9].astype(np.float)>=j*fare_bracket_size)
            & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size),1]

        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))

survival_table[survival_table != survival_table] = 0
print(survival_table)

survival_table[survival_table < 0.5] = 0
survival_table[survival_table >= 0.5] = 1

print(survival_table)

test_file = open(file_path+'test.csv','rU')
test_file_object=csv.reader(test_file)
header=test_file_object.__next__()
predictions_file=open(file_path+'genderclassmodel.csv','w', newline="")
p=csv.writer(predictions_file)
p.writerow(["PassengerId","Survived"])

for row in test_file_object:
    for j in range(number_of_price_brackets):
        try:
            row[8]=float(row[8])
        except:
            bin_fare = 3-float(row[1])
            break
        if row[8] > fare_ceiling:
            bin_fare = number_of_price_brackets - 1
            break
        if row[8] >= j*fare_bracket_size\
            and row[8] < (j+1) * fare_bracket_size:
            bin_fare=j
            break
        if row[3] == 'female':
            p.writerow([row[0], "%d" % int(survival_table[0, float(row[1])-1, bin_fare])])
        else:
            p.writerow([row[0], "%d" % int(survival_table[0, float(row[1])-1, bin_fare])])

test_file.close()
predictions_file.close()
"""