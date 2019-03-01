import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_data = []
train_inner_list = []
train_target = []

with open('datatraining.txt') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		train_inner_list.append(float(row['Temperature']))
		train_inner_list.append(float(row['Humidity']))
		train_inner_list.append(float(row['Light']))
		train_inner_list.append(float(row['CO2']))
		train_inner_list.append(float(row['HumidityRatio']))
		train_data.append(train_inner_list)
		train_inner_list = []
		train_target.append(int(row['Occupancy']))

nb = GaussianNB()
nb.fit(train_data, train_target)

rfc = RandomForestClassifier()
rfc.fit(train_data, train_target)

svm = SVC()
svm.fit(train_data, train_target)

#####for testing set 1#####
test1_data = []
test1_inner_list = []
test1_target = []

with open('datatest.txt') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		test1_inner_list.append(float(row['Temperature']))
		test1_inner_list.append(float(row['Humidity']))
		test1_inner_list.append(float(row['Light']))
		test1_inner_list.append(float(row['CO2']))
		test1_inner_list.append(float(row['HumidityRatio']))
		test1_data.append(test1_inner_list)
		test1_inner_list = []
		test1_target.append(int(row['Occupancy']))

predicted_target_nb1 = nb.predict(test1_data)
predicted_target_rfc1 = rfc.predict(test1_data)
predicted_target_svm1 = svm.predict(test1_data)

accuracy_nb1 = accuracy_score(test1_target, predicted_target_nb1)
accuracy_rfc1 = accuracy_score(test1_target, predicted_target_rfc1)
accuracy_svm1 = accuracy_score(test1_target, predicted_target_svm1)

print("\n====================================================================================")
print("For Testing Set 1")
print("====================================================================================")

print("\nAccuracy for Naïve Bayes Classifier : ")
print(accuracy_nb1)

print("\nAccuracy for Random Forest Classifier : ")
print(accuracy_rfc1)

print("\nAccuracy for Support Vector Machine Classifier : ")
print(accuracy_svm1)

#####for testing set 2#####
test2_data = []
test2_inner_list = []
test2_target = []

with open('datatest2.txt') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		test2_inner_list.append(float(row['Temperature']))
		test2_inner_list.append(float(row['Humidity']))
		test2_inner_list.append(float(row['Light']))
		test2_inner_list.append(float(row['CO2']))
		test2_inner_list.append(float(row['HumidityRatio']))
		test2_data.append(test2_inner_list)
		test2_inner_list = []
		test2_target.append(int(row['Occupancy']))

predicted_target_nb2 = nb.predict(test2_data)
predicted_target_rfc2 = rfc.predict(test2_data)
predicted_target_svm2 = svm.predict(test2_data)

accuracy_nb2 = accuracy_score(test2_target, predicted_target_nb2)
accuracy_rfc2 = accuracy_score(test2_target, predicted_target_rfc2)
accuracy_svm2 = accuracy_score(test2_target, predicted_target_svm2)

print("\n====================================================================================")
print("For Testing Set 2")
print("====================================================================================")

print("\nAccuracy for Naïve Bayes Classifier : ")
print(accuracy_nb2)

print("\nAccuracy for Random Forest Classifier : ")
print(accuracy_rfc2)

print("\nAccuracy for Support Vector Machine Classifier : ")
print(accuracy_svm2)