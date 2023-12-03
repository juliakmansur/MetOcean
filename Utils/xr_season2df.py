
import xarray as xr

def xr_season2df(ds, summer, autumn, winter, spring):
    '''
    Split Dataset acording to season:

    summer = list 
        default = [1,2,3]
    autumn = list
        default = [4,5,6]
    winter = list
        default = [7,8,9]
    sprint = list
        default = [10,11,12]

    Output:
        Dataset for each season
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

    ds_summer = ds.sel(time=ds.time.dt.month.isin(sum))
    ds_autumn = ds.sel(time=ds.time.dt.month.isin(aut))
    ds_winter = ds.sel(time=ds.time.dt.month.isin(win))
    ds_spring = ds.sel(time=ds.time.dt.month.isin(spr))

    return(ds_summer, ds_autumn, ds_winter, ds_spring)