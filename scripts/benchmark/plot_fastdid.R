library(ggplot2)
library(readr)
library(dplyr)

df <- read_csv("data/fastdid/py_benchmark.csv") |>
  select(time, order) |>
  mutate(name = "csa-py")

df_6 <- read_csv("data/fastdid/py_benchmark_6.csv") |>
  select(time, order) |>
  mutate(name = "csa-py")

df_r <- read_csv("data/fastdid/R_benchmark.csv") |>
  select(time, order) |>
  mutate(name = "fastdid")

df_r_6 <- read_csv("data/fastdid/R_benchmark_6.csv") |>
  select(time, order) |>
  mutate(name = "fastdid")

df_all <- bind_rows(df, df_6, df_r, df_r_6) |>
  mutate(
    order = as.integer(order),
    N = 10^order,
    # build parsed labels like: N==10^1, N==10^2, ...
    N_lab = factor(
      paste0("N==10^", order),
      levels = paste0("N==10^", sort(unique(order)))
    )
  )

plot_data <- function(df) {
  ggplot(df, aes(x = name, y = time, fill = name)) +
    geom_violin(trim = FALSE, alpha = 0.8, width = 2) +
    geom_boxplot(width = 0.2, outlier.shape = NA, alpha = 0.3) +
    facet_grid(~N_lab, scales = "free_y", labeller = label_parsed) +
    scale_y_log10() +
    theme_classic() +
    theme(
      text = element_text(size = 22),
      axis.text.x = element_blank(),
      axis.title.x = element_blank(),
      axis.ticks.x = element_blank(),
      strip.text.x = element_text(face = "bold")
    ) +
    labs(
      title = "Benchmark results by sample size against fastdid",
      x = "Implementation",
      y = "Runtime (log10 seconds)",
      fill = "Implementation"
    )
}

ggsave(
  "figs/bench_res_fastdid.png",
  plot = plot_data(df_all),
  width = 16,
  height = 12
)
