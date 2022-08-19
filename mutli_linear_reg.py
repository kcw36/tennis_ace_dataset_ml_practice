import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')
#print(df.head())
print(df.columns)

# perform exploratory analysis here:
#sns.pairplot(df[['FirstServe', 'Wins','Losses', 'TotalPointsWon']])
#plt.savefig('first_serve_wins_losses_total_points_pair.png')
#plt.clf()
#plt.scatter(df[['BreakPointsOpportunities']], df[['Winnings']])
# plt.scatter(df[['BreakPointsSaved']], df[['Wins']])
# plt.show()
# plt.clf()

'''
Perform single predictor variable linear regression
Repeated thrice for three different predictors
Outcome variable of Winnings or Losses
'''
#train model to predict winnings from break opportunities 
break_points = df[['BreakPointsOpportunities']]
winnings = df[['Winnings']]
break_train, break_test, winnings_train, winnings_test = train_test_split(break_points, winnings, test_size=0.2)
mlr = LinearRegression()
model = mlr.fit(break_train, winnings_train)
winnings_predicted = model.predict(break_test)

#scatter plot in subplot figure for data based on break oppurtunities model
fig1,ax = plt.subplots()
ax = plt.subplot(1, 3, 1)
plt.scatter(winnings_test, winnings_predicted, alpha=0.4)
plt.xlabel('Winnings Test Data')
plt.ylabel('Winnings Predicted Data')
plt.title('Winnings based on Break Points Opportunities')

#train model to predict losses based on double faults
double_faults = df[['DoubleFaults']]
losses = df[['Losses']]
double_train, double_test, losses_train, losses_test = train_test_split(double_faults, losses, test_size=0.2)
model2 = mlr.fit(double_train, losses_train)
losses_predicted = model2.predict(double_test)

#scatter plot in subplot figure for data based on double faults model
ax2 = plt.subplot(1,3,2)
plt.scatter(losses_test, losses_predicted, alpha=0.4)
plt.xlabel('Test Losses')
plt.ylabel('Model Losses')
plt.title('Losses based on double faults')

#train model to predict winnings based on first serve points
first_serve_points = df[['FirstServePointsWon']]
serve_train, serve_test, winnings_train, winnings_test = train_test_split(first_serve_points, winnings, test_size=0.2)
model3 = mlr.fit(serve_train, winnings_train)
winnings_predicted_serve = model3.predict(serve_test)

#scatter plot in subplot figure for data based on first serve points model
ax3 = plt.subplot(1, 3, 3, autoscale_on=True)
plt.scatter(winnings_test, winnings_predicted_serve, alpha=0.4)
plt.xlabel('Test Winnings')
plt.ylabel('Model Winnings')
plt.title('Winnings based on first serve points')

#resize figure for clarity
fig1.set_figheight(7.5)
fig1.set_figwidth(15)
plt.subplots_adjust(wspace = 0.5)

# #present and save figure
# plt.show()
# plt.savefig('Linear_Reg_examples.png')
# plt.close()
'''
Linear regression models using two predictor variables for on outcome variable
Repeated for two pairs of predictor variables
Outcome is winnings variable
'''
#train model to predict winnings from first and second serve points
first_second_serve_points = df[['FirstServePointsWon', 'SecondServePointsWon']]
first_second_train, first_second_test, winnings_train, winnings_test = train_test_split(first_second_serve_points, winnings, test_size=0.2)
model4 = mlr.fit(first_second_train, winnings_train)
winnings_predicted = mlr.predict(first_second_test)

#scatter plot for above model
fig2, ax = plt.subplots()
ax = plt.subplot(1, 2, 1)
plt.scatter(winnings_test, winnings_predicted, alpha=0.4)
plt.xlabel('Test Winnings')
plt.ylabel('Model Winnings')
plt.title('Winnings based on first and second serve points')

#train model to predict winnings from break oppurtunities and double faults
breaks_doubles = df[['BreakPointsOpportunities', 'DoubleFaults']]
bd_train, bd_test, winnings_train, winnings_test = train_test_split(breaks_doubles, winnings, test_size=0.2)
model5 = mlr.fit(bd_train, winnings_train)
winnings_predicted = mlr.predict(bd_test)

#scatter plot from above model
ax = plt.subplot(1, 2, 2)
plt.scatter(winnings_test, winnings_predicted, alpha=0.4)
plt.xlabel('Test Winnings')
plt.ylabel('Model Winnings')
plt.title('Winnings based on break opportunities and double faults')

#resize figure
fig2.set_figheight(7)
fig2.set_figwidth(10)
# #display figure
# plt.show()
# plt.close()

'''
Mutiple Linear Regression
One example
Outcome of Winnings
'''
#multi predictor model
multi_df = df[['BreakPointsOpportunities', 'DoubleFaults', 'ReturnGamesWon', 'BreakPointsSaved', 'ServiceGamesWon']]
multi_train, multi_test, winnings_train, winnings_test = train_test_split(multi_df, winnings, test_size=0.2)
model5 = mlr.fit(multi_train, winnings_train)
winnings_predicted = mlr.predict(multi_test)

#clear previous data
plt.close(fig1)
plt.close(fig2)
#setup figure
fig3, ax = plt.subplots()
#scatter for above model
ax = plt.subplot(1, 2, 1)
plt.scatter(winnings_test, winnings_predicted)
plt.xlabel('Test Winnings')
plt.ylabel('Model Winnings')
plt.title('MLR Model of Winnings')

#test model based on all predictor variables
multi_df = df[['BreakPointsOpportunities', 'DoubleFaults', 'ReturnGamesWon', 'BreakPointsSaved', 'ServiceGamesWon', 'FirstServePointsWon',\
    'FirstServeReturnPointsWon', 'SecondServePointsWon', 'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted', 'BreakPointsFaced',\
    'ReturnGamesPlayed', 'ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon', 'TotalPointsWon']]
multi_train, multi_test, winnings_train, winnings_test = train_test_split(multi_df, winnings, test_size=0.2)
model6 = mlr.fit(multi_train, winnings_train)
winnings_predicted = mlr.predict(multi_test)

#scatter for above test model
ax = plt.subplot(1, 2, 2)
plt.scatter(winnings_test, winnings_predicted)
plt.xlabel('Test Winnings')
plt.ylabel('Model Winnings')
plt.title('Test MLR Model of Winnings')

fig3.set_figheight(7)
fig3.set_figwidth(15)
plt.show()