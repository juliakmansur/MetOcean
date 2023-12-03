from urllib import request, error
from datetime import date, timedelta
from pathlib import Path

data1 = date(2016, 3, 4)
data2 = date(2020, 1, 1)
datas = [data1]

while data1 < data2:
    data1 = data1 + timedelta(days=1)
    datas.append(data1)

for i in datas:
    print(f'Starting: {i}')
    url = f"ftp://julia.kmansur@gmail.com:julia.kmansur@gmail.com@ftp.remss.com/ccmp/v03.0/daily/y{i.year}/m{i.month:02d}/CCMP_Wind_Analysis_{i.year}{i.month:02d}{i.day:02d}_V03.0_L4.0.nc"
    new = Path("../../99_testes") / f"CCMP_y{i.year}_m{i.month:02d}_d{i.day:02d}.nc"

    try:
        request.urlretrieve(url, new)
        request.urlcleanup()
        print('Done\n')
    except error.HTTPError as e:
        print(f'Error downloading {url}: {e}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
