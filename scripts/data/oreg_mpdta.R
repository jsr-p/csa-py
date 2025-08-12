library(did)
library(DRDID)

out <- att_gt(
  yname = "lemp",
  gname = "first.treat",
  idname = "countyreal",
  tname = "year",
  xformla = ~1,
  data = mpdta,
  est_method = "reg",
  bstrap = FALSE
)

jsonlite::write_json(
  as.list(data.frame(att = out$att)),
  "data/testing/att_oreg.json",
  auto_unbox = TRUE,
  digits = 16
)
write.csv(
  as.matrix(out$inffunc),
  "data/testing/if_oreg.csv",
  row.names = FALSE
)

print("Created output data")
