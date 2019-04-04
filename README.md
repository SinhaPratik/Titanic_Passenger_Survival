# Titanic_Passenger_Survival

Project- Survival Prediction for Titanic Passengers  

Dataset Used- Titanic Dataset

Training data consists of details of 891 Passengers boarding on the Titanic Ship
Prediction is implemented on a total of 418 passengers to test the accuracy of our prediction

Description about data:-
1)PassengerId: An unique index for passenger rows. It starts from 1 for first row and increments by 1 for every new rows.
2)Survived: Shows if the passenger survived or not. 1 stands for survived and 0 stands for not survived.
3)Pclass: Ticket class. 1 stands for First class ticket. 2 stands for Second class ticket. 3 stands for Third class ticket.
4)Name: Passenger's name. Name also contain title. "Mr" for man. "Mrs" for woman. "Miss" for girl. "Master" for boy.
5)Sex: Passenger's sex. It's either Male or Female.
6)Age: Passenger's age. "NaN" values in this column indicates that the age of that particular passenger has not been recorded.
7)SibSp: Number of siblings or spouses travelling with each passenger.
8)Parch: Number of parents of children travelling with each passenger.
9)Ticket: Ticket number.
10)Fare: How much money the passenger has paid for the travel journey.
11)Cabin: Cabin number of the passenger. "NaN" values in this column indicates that the cabin number of that particular passenger has not been recorded.
12)Embarked: Port from where the particular passenger was embarked/boarded.


Training data file- train.csv
Test data file- test.csv
Submissive file- submission.csv

Programming Language used- Python
Library used- Scikit-Learn

Models used- 1) Logistic Regression 
		Accuracy achieved- 81.56
	  
	     2) Support Vector Machine
		Accuracy Achieved- 82.12

The SVM model gives a better accuracy as compared to Logistic Regression so it will be beneficial to use it for predicting
a more accurate result for the given set of problem.
