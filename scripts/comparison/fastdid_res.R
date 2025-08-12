devtools::load_all("~/gh-repos/did-repos/fastdid")
library(glue)
library(jsonlite)

library(did)


data.table::setDTthreads(0)

run_fastdid <- function(dt) {
  return(result)
}

order <- 4
dt <- read.csv(glue("data/fastdid/sim{order}.csv"))

res <- fastdid(
  dt,
  timevar = "time",
  cohortvar = "G",
  unitvar = "unit",
  outcomevar = "y",
  result_type = "group_time",
  copy = FALSE,
  allow_unbalance_panel = FALSE,
  base_period = "varying"
)


dfres <- data.frame(
  att = res$att,
  se = res$se,
  g = res$cohort,
  t = res$time
) |> dplyr::arrange(g, t)

write.csv(dfres, file = "data/testing/fastdid-fd.csv", row.names = FALSE)



# CSA
out <- att_gt(
  yname = "y",
  gname = "G",
  idname = "unit",
  tname = "time",
  control_group = c("notyettreated"),
  allow_unbalanced_panel = FALSE,
  data = dt,
  est_method = "dr",
  bstrap = FALSE
)



rescsa <- data.frame(
  att = out$att,
  se = out$se,
  g = out$group,
  t = out$t
)
write.csv(rescsa, "data/testing/fastdid-csa.csv", row.names = FALSE)
