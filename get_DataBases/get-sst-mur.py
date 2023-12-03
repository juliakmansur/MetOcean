#!/usr/bin/env python3

from datetime import date, timedelta
import netCDF4
import xarray as xa

# Função para baixar e processar dados
def process_data(start_date, end_date, lat_range=(-14.3, -4.3), lon_range=(-39.1, -33)):
    datas = [start_date]
    
    # Gerar lista de datas
    while start_date < end_date:
        start_date = start_date + timedelta(days=1)
        datas.append(start_date)

    # Iterar sobre as datas
    for i in datas:
        try:
            # Abrir o conjunto de dados a partir da URL
            url = f'https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L4/GLOB/JPL/MUR/v4.1/{i.year}/{i.timetuple().tm_yday.zfill(3)}/{i.year}{i.month.zfill(2)}{i.day.zfill(2)}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'
            arquivo = xa.open_dataset(url, decode_times=False)
            
            print(str(i))

            # Selecionar subconjunto de dados e salvar em um novo arquivo
            sst = arquivo['analysed_sst'].sel(lat=slice(*lat_range), lon=slice(*lon_range))
            sst.to_netcdf(f'../../Dados/MUR/MUR_{i.year}/{i}.nc')
            
            print('Done')
        except Exception as e:
            # Lidar com erros
            print(f"Error processing {i}: {e}")

if __name__ == "__main__":
    # Definir datas de início e fim
    data_ini = date(2006, 6, 3)
    data_fim = date(2006, 6, 3)

    # Definir intervalos de latitude e longitude
    latitude_range = (-14.3, -4.3)
    longitude_range = (-39.1, -33)

    # Chamar a função de processamento
    process_data(data_ini, data_fim, lat_range=latitude_range, lon_range=longitude_range)
