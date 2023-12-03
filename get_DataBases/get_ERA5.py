import cdsapi

years = list(map(str, range(2002, 2020)))
months = [f"{month:02d}" for month in range(1, 13)]
days = [f"{day:02d}" for day in range(1, 32)]
hours = [f"{hour:02d}:00" for hour in range(24)]

area = [-4.3, -39.1, -14.3, -33]
output_folder = '../Dados/ERA5/'

c = cdsapi.Client()

for year in years:
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'grib',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                    'mean_sea_level_pressure', 'sea_surface_temperature', 'surface_pressure',
                ],
                'year': year,
                'month': months,
                'day': days,
                'time': hours,
                'area': area,
            },
            f'{output_folder}{year}.grib'
        )
        print(f'Data for {year} downloaded successfully.')
    except Exception as e:
        print(f'Error downloading data for {year}: {e}')
