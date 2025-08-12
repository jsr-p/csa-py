# library(did)
devtools::load_all("/home/jsr-p/gh-repos/did-repos/fork-did-csa")

extract <- function(es) {
  list(
    res = data.frame(
      e = es$egt,
      att = es$att,
      se = es$se.egt
    ),
    oatt = es$overall.att,
    ose = es$overall.se
  )
}

out <- att_gt(
  yname = "lemp",
  gname = "first.treat",
  idname = "countyreal",
  tname = "year",
  xformla = ~1,
  data = mpdta,
  est_method = "dr",
  bstrap = FALSE
)

es <- aggte(out, type = "group")
group <- extract(es)
es <- aggte(out, type = "dynamic")
print(es)
dynamic <- extract(es)
es <- aggte(out, type = "calendar")
calendar <- extract(es)
es <- aggte(out, type = "simple")
simple <- extract(es)

res <- list(
  dynamic = dynamic,
  group = group,
  calendar = calendar,
  simple = simple
)
jsonlite::write_json(
  res,
  "data/testing/res.json",
  auto_unbox = TRUE,
  digits = 16
)

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
  as.list(data.frame(att = out$att, se = out$se)),
  "data/testing/att_oreg.json",
  auto_unbox = TRUE,
  digits = 16
)

print("Created output data")


# oreg res output aggregations

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
es <- aggte(out, type = "group")
group <- extract(es)
es <- aggte(out, type = "dynamic")
print(es)
dynamic <- extract(es)
es <- aggte(out, type = "calendar")
calendar <- extract(es)
es <- aggte(out, type = "simple")
simple <- extract(es)
res <- list(
  dynamic = dynamic,
  group = group,
  calendar = calendar,
  simple = simple
)
jsonlite::write_json(
  res,
  "data/testing/resoreg.json",
  auto_unbox = TRUE,
  digits = 16
)
print("Created oreg output data")


# oreg res output aggregations

out <- att_gt(
  yname = "lemp",
  gname = "first.treat",
  idname = "countyreal",
  tname = "year",
  xformla = ~lpop,
  data = mpdta,
  est_method = "reg",
  bstrap = FALSE
)
es <- aggte(out, type = "group")
group <- extract(es)
es <- aggte(out, type = "dynamic")
print(es)
dynamic <- extract(es)
es <- aggte(out, type = "calendar")
calendar <- extract(es)
es <- aggte(out, type = "simple")
simple <- extract(es)
res <- list(
  dynamic = dynamic,
  group = group,
  calendar = calendar,
  simple = simple
)
jsonlite::write_json(
  res,
  "data/testing/resoregxvars.json",
  auto_unbox = TRUE,
  digits = 16
)
print("Created oreg output data")
