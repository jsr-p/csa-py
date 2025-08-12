library(DRDID)

dir.create("data/testing", showWarnings = TRUE)

# Form the Lalonde sample with CPS comparison group (data in wide format)
eval_lalonde_cps <- subset(nsw, nsw$treated == 0 | nsw$sample == 2)
# Further reduce sample to speed example
set.seed(123)
unit_random <- sample(1:nrow(eval_lalonde_cps), 5000)
eval_lalonde_cps <- eval_lalonde_cps[unit_random, ]
# Select some covariates
covX <- as.matrix(cbind(
  1, eval_lalonde_cps$age, eval_lalonde_cps$educ,
  eval_lalonde_cps$black, eval_lalonde_cps$married,
  eval_lalonde_cps$nodegree, eval_lalonde_cps$hisp,
  eval_lalonde_cps$re74
))

y1 <- eval_lalonde_cps$re78
y0 <- eval_lalonde_cps$re75
D <- eval_lalonde_cps$experimental
covariates <- covX

df <- data.frame(
  y1 = y1,
  y0 = y0,
  D = D,
  covariates = covariates
)

# export test data
write.csv(df, file = "data/testing/lalonde_cps.csv", row.names = FALSE)

# Implement traditional DR locally efficient DiD with panel data
out <- drdid_panel(
  y1 = y1,
  y0 = y0,
  D = D,
  covariates = covariates,
  boot = FALSE,
  inffunc = TRUE
)

jsonlite::write_json(
  list(
    att = out$ATT,
    se = out$se,
    IF = out$att.inf.func
  ),
  "data/testing/dr_panel.json",
  auto_unbox = TRUE,
  digits = 16
)
