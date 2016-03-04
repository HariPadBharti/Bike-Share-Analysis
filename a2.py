import matplotlib.pyplot as plt
import datetime as datetime
import pandas.io.data
import csv
import pandas as pd
from sklearn import tree
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
def aggregate_bike_data(data_filenames):
    df1=pd.DataFrame()
    df=pd.DataFrame()
    df2=pd.DataFrame()
    list=[]
    for filename in data_filenames:
        df=pd.read_csv(filename,usecols=[1])
        list.append(df)
    df1=pd.concat(list)
    pd.to_datetime(df1['Start date'])
    df1['Start date'] = df1['Start date'].apply(lambda x: pd.datetools.parse(x).strftime('%Y%m%d'))
    df1['Counts'] = df1.groupby(['Start date'])['Start date'].transform('count')
    df1=df1.drop_duplicates(subset='Start date', take_last = True)
    df2=pd.DataFrame(df1)
    df2.columns = ['DATE', 'Count']
    df2.index = range(0,len(df1))
    #print(df2.DATE.type)
    #df2.to_csv('out.csv')
    #Returning the modified dataframe having date in format YYYYMMDD with unique dates. 
    return df2
    
def integrate_weather_data(df, weather_filename):
    df3=pd.DataFrame()
    df2=pd.read_csv(weather_filename,index_col=None,header=0)
    df2=df2.drop(df2.columns[[0, 1]],axis=1) 
    
    df2['DATE'] = [datetime.datetime.strptime(str(date_val),'%Y%m%d').strftime('%Y%m%d') for date_val in df2['DATE']]

    #df2.index = range(1,len(df) + 1)
    #print(df2)
    #df3=df['Count']
    #print(df2.head())
    #print(df.head())
    
    df3 = pd.merge(df, df2, on='DATE', how='right')

    #df2.index = range(1,len(df) + 1)
    #print(df.DATE.dtype)
    #print(df3)
    # load the weather data into a new dataframe and
    # merging the weather data and the bike sharing data together (by date)
    # returning a single dataframe that contains all the data with count of bikeshares and 

    return df3
def create_month_plot(df):
    df['DATE'] = [datetime.datetime.strptime(str(date_val),'%Y%m%d').strftime('%m') for date_val in df['DATE']]
    #print(df['DATE'])
    df=df.groupby(['DATE'])['Count'].sum()
    #print(df)
    df.plot(kind='bar')
    plt.xlabel("MONTH")
    plt.ylabel("No of Bikeshares")

    

    # created a bar chart showing the number of bikeshares per month
    
    # nothing to return

def create_scatterplot(df):
    x=df['TMAX']
    y=df['Count']
    plt.scatter(x,y)
    plt.xlabel("TMAX")
    plt.ylabel("No of Bikeshares")
    # create a scatterplot showing the number of bikeshares versus the maximum temperature
    # do not display this here (run does this)
    # nothing to return
    
def run_regression(df):
    y=df['Count']
    x=df.ix[:,2:]
    clf = tree.DecisionTreeRegressor()
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    
    clf.fit(X_train, y_train)
    #X.iloc[X_train]
    
    #print(X_test)
    #print(clf.predict(X_test))
    #train, test = train_test_split(df, test_size = 0.2)
    
    
    # score is on the testing data
    return clf.score(X_test, y_test)
def run():
    fnames = ["2014-Q1-Trips-History-Data2.csv",
              "2014-Q2-Trips-History-Data2.csv",
              "2014-Q3-Trips-History-Data3.csv",
              "2014-Q4-Trips-History-Data.csv"]
    df = aggregate_bike_data(fnames)
    print(df.head(10))
    wdf = integrate_weather_data(df, "dc-weather-2014.csv")
    print(wdf.head(10))
    create_month_plot(wdf)
    plt.show() # shows the plot on screen
    create_scatterplot(wdf)
    plt.show() # shows the plot on screen
    print("The plots show that the bikeshares are maximum during the months of may to october having temperature ranging between 225 - 325,  with July having maximum bikeshares. Noticable part is the temperature range of 225-325 that comes with maximum bike shares that probably suggests months b/w May and October ") 
    print("Score after running regression:", run_regression(wdf))
    print("The key attributes used by the predictor are:TMAX, PRCP, AWND etc") # ***Complete***

if __name__ == '__main__':
    run()

    
