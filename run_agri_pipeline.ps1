python -m venv .venv
. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install pandas requests openpyxl

if (Test-Path .\get_comtrade_and_psd.R) {
    if (Get-Command Rscript -ErrorAction SilentlyContinue) {
        Rscript .\get_comtrade_and_psd.R
    } else {
        Write-Host "Rscript not found; skipping UN Comtrade step."
    }
}

python .\fill_agri_from_sources.py --input .\agri_input.csv --outbase agri_filled

Write-Host "`n=============================="
Write-Host "DONE. Output files:"
Write-Host "  agri_filled.csv"
Write-Host "  agri_filled.xlsx"
Write-Host "=============================="
