#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(comtradr)
  library(dplyr)
  library(readr)
})

reporters <- c("USA", "CAN", "AUS", "RUS", "UKR", "IND", "ARG", "BRA", "KAZ")
start_year <- 2000
end_year <- 2024

message("Fetching UN Comtrade wheat export data ...")
exports <- ct_get_data(
  trade_flow = "X",
  reporters = reporters,
  partners = "World",
  start_date = start_year,
  end_date = end_year,
  commod_codes = "1001",
  freq = "A"
)

exports <- exports %>%
  mutate(netweight_tonnes = netweight_kg / 1000) %>%
  select(area_code = reporter_iso, year, wheat_exports_comtrade = netweight_tonnes)

write_csv(exports, "comtrade_wheat_exports.csv")

zip_path <- "psd.zip"
if (file.exists(zip_path)) {
  message("Parsing PSD zip ...")
  tmpdir <- tempdir()
  utils::unzip(zip_path, exdir = tmpdir)
  psd_file <- list.files(tmpdir, pattern = "\\.csv$", full.names = TRUE)[1]
  if (!is.na(psd_file)) {
    psd <- read_csv(psd_file, show_col_types = FALSE)
    subset <- psd %>%
      filter(commodity %in% c("WHEAT", "BARLEY", "CORN", "SOYBEANS")) %>%
      select(area_code = Country_Code, year = Market_Year, commodity, production = Value)
    write_csv(subset, "psd_production_subset.csv")
  }
}

message("Done.")
