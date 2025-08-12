library(did)
library(jsonlite)


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

res <- did::mboot(
  out$inffunc,
  out$DIDparams
)

out_test <- list(
  IF = as.matrix(out$inffunc),
  bres = as.list(as.data.frame(res$bres)),
  V = as.list(as.data.frame(res$V)),
  se = as.list(res$se),
  crit.val = res$crit.val,
  bSigma = as.list(as.data.frame(res$bSigma)),
  att = out$att
)

write_json(
  out_test,
  "data/testing/bstrap_data.json",
  auto_unbox = TRUE,
  digits = 16
)


aggte(
  out,
  type = "group",
  bstrap = TRUE,
)
