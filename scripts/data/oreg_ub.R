library(did)
library(DRDID)
devtools::load_all("/home/jsr-p/gh-repos/did-repos/did")

df <- read.csv("data/testing/sim_ub.csv")
print(df |> head())
print(df |> dim())
df <- dplyr::rename(df, year = t) # tname equals to `t` bugs out
out <- att_gt(
  yname = "y",
  gname = "g",
  idname = "id",
  tname = "year",
  control_group = c("notyettreated"),
  allow_unbalanced_panel = TRUE,
  xformla = ~X,
  data = df,
  est_method = "reg",
  bstrap = FALSE
)

write.csv(
  data.frame(out$att, out$se),
  "data/testing/resoregub.csv",
  row.names = FALSE
)
write.csv(
  out$inffunc |> as.matrix(),
  "data/testing/if_oreg_all_new.csv",
  row.names = FALSE
)

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

es <- aggte(out, type = "simple", na.rm = TRUE)
esg <- aggte(out, type = "group", na.rm = TRUE)
esc <- aggte(out, type = "calendar", na.rm = TRUE)
esd <- aggte(out, type = "dynamic", na.rm = TRUE)

simple <- extract(es)
group <- extract(esg)
dynamic <- extract(esd)
calendar <- extract(esc)

res <- list(
  dynamic = dynamic,
  group = group,
  calendar = calendar,
  simple = simple
)
jsonlite::write_json(
  res,
  "data/testing/resoregunbalanced.json",
  auto_unbox = TRUE,
  digits = 16
)
print("Created oreg output data")


# Export influence functions

simple <- data.frame(
  simple_att = es$inf.function$simple.att
)

group <- list(
  select = data.frame(
    selective_inf_func_g = esg$inf.function$selective.inf.func.g
  ),
  selective = data.frame(
    selective_inf_func = esg$inf.function$selective.inf.func
  )
)

calendar <- list(
  calendar = data.frame(
    calendar_inf_func = esc$inf.function$calendar.inf.func
  ),
  calendar_t = data.frame(
    calendar_inf_func_t = esc$inf.function$calendar.inf.func.t
  )
)

dynamic <- list(
  dynamic = data.frame(
    dynamic_inf_func = esd$inf.function$dynamic.inf.func
  ),
  dynamic_e = data.frame(
    dynamic_inf_func_e = esd$inf.function$dynamic.inf.func.e
  )
)

res <- list(
  simple = simple,
  group = group,
  calendar = calendar,
  dynamic = dynamic
)

jsonlite::write_json(
  res,
  "data/testing/resoreguballifs.json",
  dataframe = "columns",
  auto_unbox = TRUE,
  digits = 16
)

print("Created oreg output data")
