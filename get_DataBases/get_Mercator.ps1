Function GetMercator {
    Param(
        [double]$lonmin, [double]$lonmax, [double]$latmax, [double]$latmin,
        [datetime]$startDate, [int]$daysToSkip, [datetime]$endDate,
        [string]$dire, [string]$filename, [switch]$reanalysis, 
        [string]$user, [securestring]$password
    )

    # Exibir informações sobre o período de contagem
    Write-Host "Count forward from:" $startDate.ToString("yyyy-MM-dd HH:mm:ss")    
    Write-Host "Count forward until:" $endDate.ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host "Count every" $daysToSkip "day(s)"

    $lon1 = $lonmin
    $lon2 = $lonmax
    $lat1 = $latmax
    $lat2 = $latmin

    while ($startDate -le $endDate) {
        # Exibir a data atual
        Write-Host $startDate.ToString("yyyy-MM-dd HH:mm:ss")
        
        $startDate1 = $startDate.ToString("yyyy-MM-dd HH:mm:ss")
        $startDate2 = $startDate.AddDays($daysToSkip).ToString("yyyy-MM-dd HH:mm:ss")

        $dateName = $startDate.ToString("yyyyMMdd")
        $fname = "$filename$dateName.nc"

        # Converter SecureString para String
        $plainPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($password))

        $pythonCommand = if ($reanalysis) {
            # Comando Python para reanálise
            "python -m motuclient --motu http://my.cmems-du.eu/motu-web/Motu --service-id GLOBAL_REANALYSIS_PHY_001_030-TDS --product-id global-reanalysis-phy-001-030-daily `
            --longitude-min $lon1 --longitude-max $lon2 --latitude-min $lat1 --latitude-max $lat2 `
            --date-min $startDate1 --date-max $startDate2 `
            --depth-min 0.493 --depth-max 318.1274 `
            --variable bottomT --variable mlotst --variable so --variable thetao --variable uo --variable vo --variable zos `
            --out-dir $dire --out-name $fname `
            --user '$user' --pwd '$plainPassword'"
        } else {
            # Comando Python para análise global
            "python -m motuclient --motu http://nrt.cmems-du.eu/motu-web/Motu --service-id GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS --product-id global-analysis-forecast-phy-001-024 `
            --longitude-min $lon1 --longitude-max $lon2 --latitude-min $lat1 --latitude-max $lat2 `
            --date-min $startDate1 --date-max $startDate2 `
            --depth-min 0.493 --depth-max 318.1274 `
            --variable bottomT --variable mlotst --variable so --variable thetao --variable uo --variable vo --variable zos `
            --out-dir $dire --out-name $fname `
            --user '$user' --pwd '$plainPassword'"
        }

        try {
            # Executar o comando Python
            Invoke-Expression -Command $pythonCommand
            Write-Host 'Data downloaded successfully.'
        } catch {
            # Lidar com erros durante o download
            Write-Host "Error downloading data: $_"
        }

        # Avançar para a próxima data
        $startDate = $startDate.AddDays($daysToSkip)
    }
}

# Definir a data inicial e final
$dataini = "2019-12-31 12:00:00"
$datafim = "2019-12-31 12:00:00"

# Solicitar nome de usuário e senha do usuário
$user = Read-Host -Prompt "Digite seu nome de usuário"
$password = Read-Host -Prompt "Digite sua senha" -AsSecureString

# Chamada da função GetMercator com os parâmetros fornecidos
GetMercator -lonmin '-39.1' -lonmax '-33' -latmax '-4.3' -latmin '-14.3' `
    -startDate $dataini -daysToSkip 1 -endDate $datafim `
    -dire "../Dados/Mercator/" -filename "Merc_" -reanalysis -user $user -password $password
