library(microbenchmark)
devtools::load_all("~/gh-repos/did-repos/fastdid")
library(glue)
library(dplyr)

data.table::setDTthreads(0)

run_fastdid <- function(dt) {
  result <- fastdid(
    dt,
    timevar = "time",
    cohortvar = "G",
    unitvar = "unit",
    outcomevar = "y",
    result_type = "group_time",
    copy = FALSE,
    allow_unbalance_panel = FALSE
  )
}


run_benchmark <- function(
    min_order,
    max_order,
    times = 25L) {
  all_bm <- list()
  for (order in min_order:max_order) {
    dt <- read.csv(glue("data/fastdid/sim{order}.csv"))
    ct <- microbenchmark(
      fastdid = {
        fastdid <- run_fastdid(dt)
      },
      times = times
    )
    df <- data.frame(ct) |>
      mutate(time = time / 1e9) |>
      rename(name = expr) |>
      select(name, time) |>
      mutate(
        n = order,
        order = order,
        lang = "R"
      )
    all_bm[[order]] <- df
    cat("Finished order", order, "\n")
    cat("Summary list:", "\n")
  }
  return(all_bm)
}

min_order <- 2
max_order <- 5
bm_result <- run_benchmark(min_order, max_order, times = 25L)
res <- do.call(rbind, bm_result)
write.csv(res, "data/fastdid/R_benchmark.csv")

bm_result <- run_benchmark(6, 6, times = 10L)
res <- do.call(rbind, bm_result)
write.csv(res, "data/fastdid/R_benchmark_6.csv")
