library(glue)
devtools::load_all("~/gh-repos/did-repos/fastdid")

# fastdid doesn't allow for DR when having unbalanced panel; a common case in
# applied work
# this script

opt <- list(N = 2500, delta = 6, e = 15)
N <- opt$N
delta <- opt$delta
e <- opt$e

# data/testing/sim_ub_10000_15_3.csv
file <- glue("data/testing/sim_ub_{N}_{e}_{delta}.csv")
df <- read.csv(file) |>
  dplyr::rename(year = t)

res <- fastdid(
  df |> as.data.table(),
  timevar = "year",
  cohortvar = "g",
  unitvar = "id",
  outcomevar = "y",
  result_type = "group_time",
  control_option = "notyet",
  control_type = "dr",
  allow_unbalance_panel = TRUE,
  base_period = "varying",
  covariatesvar = "X"
)

# Yields:
# Error in validate_argument(dt, p) :
#   fastdid does not support DR when allowing for unbalanced panels.
# Calls: fastdid -> validate_argument
# Execution halted
