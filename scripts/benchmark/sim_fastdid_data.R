library(glue)
devtools::load_all("~/gh-repos/did-repos/fastdid")

min_order <- 2
max_order <- 6

dir.create("data/fastdid")

for(order in min_order:max_order){
  dt <- sim_did(
    10^order,
    10,
    seed = 1,
    cov = "cont",
    second_outcome = TRUE,
    second_cov = TRUE
  )[["dt"]]

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

  outfile <- glue("data/fastdid/sim{order}.csv")
  write.csv(dt, outfile, row.names = FALSE)

  cat(paste("Saved to", outfile, "\n"))
}
