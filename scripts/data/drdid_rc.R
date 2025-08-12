devtools::load_all("/home/jsr-p/gh-repos/did-repos/DRDID")

# use the simulated data provided in the package

sim_rc |> head()

covX <- as.matrix(cbind(1, sim_rc[, 5:8]))

# Implement the 'traditional' locally efficient DR DiD estimator
out <- drdid_rc(
  y = sim_rc$y, post = sim_rc$post, D = sim_rc$d,
  covariates = covX
)
print(out)

write.csv(sim_rc, file = "data/testing/sim_rc.csv", row.names = FALSE)
