
import pandas as pd

def pd_season(df,  summer, autumn, winter, spring):
    '''
    Split Dataframe acording to season:

    summer = list 
        default = [1,2,3]
    autumn = list
        default = [4,5,6]
    winter = list
        default = [7,8,9]
    sprint = list
        default = [10,11,12]

    Output:
        Dataframe for each season
        summer, autumn, winter, spring

    '''
    sum = [1,2,3]
    if summer:
        sum = summer

    aut = [4,5,6]
    if autumn:
        aut = autumn
    
    win = [7,8,9]
    if winter:
        win = winter
    
    spr = [10,11,12]
    if spring:
        spr = spring


    df_summer = df[df.index.month.isin(sum)]
    df_autumn = df[df.index.month.isin(aut)]
    df_winter = df[df.index.month.isin(win)]
    df_spring = df[df.index.month.isin(spr)]

    return(df_summer, df_autumn, df_winter, df_spring)