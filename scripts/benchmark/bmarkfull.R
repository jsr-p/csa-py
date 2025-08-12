#!/usr/bin/env Rscript

library(microbenchmark)
library(jsonlite)
library(glue)
library(dplyr)
library(optparse)
library(did)
library(dplyr)
devtools::load_all("~/gh-repos/did-repos/fastdid")

option_list <- list(
  make_option(c("-N", "--N"), type = "numeric", help = "Sample size"),
  make_option(c("-d", "--delta"), type = "numeric", help = "Delta value"),
  make_option(c("-e", "--e"), type = "numeric", help = "Number of time periods")
)

opt <- parse_args(OptionParser(option_list = option_list))

N <- opt$N
delta <- opt$delta
e <- opt$e

# data/testing/sim_ub_10000_15_3.csv
file <- glue("data/testing/sim_ub_{N}_{e}_{delta}.csv")
df <- read.csv(file) |>
  dplyr::rename(year = t)

fn <- function() {
  out <- att_gt(
    yname = "y",
    gname = "g",
    idname = "id",
    tname = "year",
    control_group = "notyettreated",
    allow_unbalanced_panel = TRUE,
    xformla = ~X,
    data = df,
    est_method = "dr",
    bstrap = FALSE,
    cband = FALSE
  )
  esd <- aggte(out, type = "dynamic", na.rm = TRUE)
}

cat(glue("Profiling att_gt with N={N}, e={e}, delta={delta}\n"))

ct <- microbenchmark(
  fastdid = {
    fastdid <- fn()
  },
  times = times
)
df <- data.frame(ct) |>
  mutate(time = time / 1e9) |>
  rename(name = expr) |>
  select(name, time) |>
  mutate(
    lang = "R",
    N = N,
    delta = delta,
    e = e,
    file = file
  )
write.csv(
  df,
  glue("data/benchmark/R_{N}_{e}_{delta}.json")
)
