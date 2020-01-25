library(tidymodels)
library(tidyverse)
library(tune)


library(DALEX)
library(iBreakDown)
library(ingredients)

library(scales)
library(lemon)
library(tidylog)

set.seed(42)

# 可視化用theme -------------------------------------------------------------------

cols = c(rgb(0, 113, 188, maxColorValue = 255),
         rgb(255, 80, 80, maxColorValue = 255),
         rgb(55, 55, 55, maxColorValue = 255))



base_family =  "Noto Sans JP Regular"
bold_family =  "Noto Sans JP Medium"

theme_line = function() {
  theme_minimal(base_size = 12, base_family = base_family) %+replace% 
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(color = "gray", size = 0.1),
          axis.title = element_text(size = 15, color = cols[3]),
          axis.title.x = element_text(margin = margin(10, 0, 0, 0), hjust = 1),
          axis.title.y = element_text(margin = margin(0, 10, 0, 0), angle = 90, hjust = 1),
          axis.text = element_text(size = 12, color = cols[3]),
          axis.line.x.bottom = element_line(color = cols[3], size = 0.5),
          axis.ticks.x = element_line(color = cols[3], size = 0.5),
          axis.ticks.length.x = unit(5, units = "pt"),
          strip.text = element_text(size = 15, color = cols[3], margin = margin(5, 5, 5, 5)),
          plot.title = element_text(size = 15, color = cols[3], margin = margin(5, 0, 5, 0)),
          plot.subtitle = element_text(size = 15, hjust = -0.05, color = cols[3],
                                       margin = margin(5, 5, 5, 5)),
          legend.position = "none"
    )
}


theme_bar = function() {
  theme_minimal(base_size = 12, base_family = base_family) %+replace%
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.major.x = element_line(color = "gray", size = 0.1),
          axis.title = element_text(size = 15, color = cols[3]),
          axis.title.x = element_text(margin = margin(10, 0, 0, 0), hjust = 1),
          axis.title.y = element_text(margin = margin(0, 10, 0, 0), angle = 90, hjust = 1),
          axis.text = element_text(size = 12, color = cols[3]),
          strip.text = element_text(size = 15, color = cols[3], margin = margin(5, 5, 5, 5)),
          plot.title = element_text(size = 15, color = cols[3], margin = margin(5, 0, 5, 0)),
          plot.subtitle = element_text(size = 15, hjust = -0.05, color = cols[3],
                                       margin = margin(5, 5, 5, 5)),
          legend.position = "none"
    )
}


theme_scatter = function() {
  theme_minimal(base_size = 12, base_family = base_family) %+replace%
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(color = "gray", size = 0.1),
          axis.title = element_text(size = 15, color = cols[3]),
          axis.title.x = element_text(margin = margin(10, 0, 0, 0), hjust = 1),
          axis.title.y = element_text(margin = margin(0, 10, 0, 0), angle = 90, hjust = 1),
          axis.text = element_text(size = 12, color = cols[3]),
          # axis.line.x.bottom = element_line(color = cols[3], size = 0.5),
          # axis.ticks.x = element_line(color = cols[3], size = 0.5),
          # axis.ticks.length.x = unit(5, units = "pt"),
          strip.text = element_text(size = 15, color = cols[3], margin = margin(5, 5, 5, 5)),
          plot.title = element_text(size = 15, color = cols[3], margin = margin(5, 0, 5, 0)),
          plot.subtitle = element_text(size = 15, hjust = -0.05, color = cols[3],
                                       margin = margin(5, 5, 5, 5)),
          legend.position = "none")
}

save_plot = function(fname, width = 8, height = 4) {
  ggsave(fname, width = width, height = height, dpi = "retina")
}

# main --------------------------------------------------------------------
df = diamonds %>% 
  mutate(log_price = log(price)) %>% 
  select(log_price, carat, cut, color, clarity, depth, table)

df %>% 
  write_csv("data/sample_data.csv")

# model -------------------------------------------------------------------

model = parsnip::rand_forest(mode = "regression",
                             trees = 500,
                             min_n = 5) %>%
  parsnip::set_engine(engine = "ranger",
                      num.threads = parallel::detectCores(),
                      seed = 42)

model_trained = model %>%
  parsnip::fit(log_price ~ ., data = df)

model_trained

# cross validation -------------------------------------------------------------------
# 
# df_cv = rsample::vfold_cv(df, v = 10)
# 
# rec = recipes::recipe(price ~ ., data = df) %>% 
#   recipes::step_ordinalscore(all_nominal())
# 
# 

# 
# 
# df_cv_result = tune::fit_resamples(rec, 
#                                    model = model,
#                                    resamples = df_cv,
#                                    metrics = metric_set(rsq, rmse, mae),
#                                    control = control_resamples(verbose = TRUE))
# 
# 
# df_cv_result %>% 
#   tune::collect_metrics()


# DALEX explainer -----------------------------------------------------------------



explainer = model_trained %>% 
  DALEX::explain(data = df %>% select(-log_price),
                 y = df %>% pull(log_price), 
                 label = "Random Forest")

explainer

# Permutation Feature Importance ------------------------------------------

pfi = explainer %>% 
  ingredients::feature_importance(type = "ratio", B = 1)

plot(pfi)

df_pfi = pfi %>% 
  as_tibble() %>% 
  group_by(variable) %>% 
  summarise(dropout_loss = mean(dropout_loss)) %>% 
  filter(!variable %in% c("_full_model_", "_baseline_")) 
  
df_pfi %>%   
  ggplot(aes(fct_reorder(variable, dropout_loss), dropout_loss)) +
  geom_col(fill = cols[1], width = 0.8, alpha = 0.9) +
  coord_flip() +
  scale_y_continuous(breaks = breaks_width(2),
                     labels = label_percent(1, big.mark = ",")) + 
  labs(x = NULL, y = "Dropout Loss") + 
  theme_bar()

save_plot("figure/pfi.png")




# Partial Dependence -------------------------------------------------

pd_num = explainer %>% 
  ingredients::partial_dependency(variable_type = "numerical")

pd_cat = explainer %>% 
  ingredients::partial_dependency(variable_type = "categorical")


plot(pd_cat)

df_pd_num = pd_num %>% 
  as_tibble() %>% 
  select(var_name = `_vname_`, x = `_x_`, y = `_yhat_`) %>% 
  mutate(var_name  = as.character(var_name))


df_pd_cat = pd_cat %>% 
  as_tibble() %>% 
  select(var_name = `_vname_`, x = `_x_`, y = `_yhat_`) %>% 
  mutate(var_name  = as.character(var_name))


df_pd_num %>% 
  ggplot(aes(x, y)) +
  geom_line(size = 1, color = cols[1]) +
  facet_rep_wrap(~var_name, scales = "free_x", repeat.tick.labels = "all") + 
  labs(x = NULL, y = "Average Prediction") + 
  theme_scatter()

save_plot("figure/pd_num.png", width = 16, height = 6)


df_pd_cat %>% 
  ggplot(aes(fct_rev(x), y)) +
  geom_col(fill = cols[1], width = 0.8, alpha = 0.8) +
  coord_flip() + 
  facet_rep_wrap(~var_name, scales = "free_y", repeat.tick.labels = "all") + 
  labs(x = NULL, y = "Average Prediction") + 
  scale_y_continuous(labels = label_number(1),
                     expand = expand_scale(0, 0.2)) + 
  theme_scatter() +
  theme(panel.grid.major.y = element_blank())

save_plot("figure/pd_cat.png", width = 16, height = 6)



df_pd_cat %>%
  filter(var_name == "clarity") %>% 
  ggplot(aes(x, y)) +
  geom_col(width = 0.8, fill = cols[1], alpha = 0.8) +
  labs(x = "clarity", y = "Average Prediction") +
  scale_y_continuous(labels = label_number(1)) +
  theme_scatter() +
  theme(panel.grid.major.x = element_blank())

save_plot("figure/pd_clarity.png", width = 6, height = 4)

df %>% 
  ggplot(aes(clarity, log_price)) +
  geom_boxplot(alpha = 0.3, color = cols[3], fill = cols[1]) +
  theme_scatter() +
  theme(panel.grid.major.x = element_blank())

save_plot("figure/boxplot_clarity.png", width = 6, height = 4)

df %>% 
  sample_n(10000) %>% 
  ggplot(aes(carat, log_price)) +
  geom_point(alpha = 0.1, color = cols[3]) +
  geom_smooth(size = 1, color = cols[1], fill = cols[1], alpha = 0.2) +
  scale_x_continuous(breaks = breaks_width(1)) + 
  theme_scatter()

save_plot("figure/carat.png", width = 6, height = 4)




df %>% 
  ggplot(aes(clarity, log_price)) +
  geom_boxplot(alpha = 0.3, color = cols[3], fill = cols[1]) +
  theme_scatter() +
  theme(panel.grid.major.x = element_blank())
  
save_plot("figure/clarity_log_price.png", width = 4, height = 4)


df %>% 
  ggplot(aes(clarity, carat)) +
  geom_boxplot(alpha = 0.3, color = cols[3], fill = cols[1]) +
  theme_scatter() +
  theme(panel.grid.major.x = element_blank())

save_plot("figure/clarity_carat.png", width = 4, height = 4)





# Individual Conditional Expectation --------------------------------------

df_instance = df %>% slice(4200) %>% as.data.frame()

ice = explainer %>% 
  ingredients::ceteris_paribus(new_observation = df_instance)

plot(ice, variables = "carat") +
  show_observations(ice, variables = "carat")

df_ice = ice %>% 
  as_tibble() %>% 
  rename(var_name = `_vname_`,  y = `_yhat_`, id = `_ids_`) 

df_pred = model_trained %>% 
  predict(df_instance) %>% 
  bind_cols(df_instance)

df_ice %>% 
  filter(var_name == "carat") %>% 
  ggplot(aes(carat, y)) +
  geom_line(aes(group = id), size = 1, color = cols[1]) +
  geom_point(aes(carat, .pred), data = df_pred, 
             size = 4, shape = 21, color = "white", fill = cols[2]) + 
  scale_x_continuous(breaks = breaks_width(1)) + 
  labs(x = "carat", y = "log_price") + 
  theme_scatter()


save_plot("figure/ice_carat.png")




# PD with Many ICE ----------------------------------------------------------------

df_subset = df %>% 
  sample_n(100) %>%
  as.data.frame()

ices = explainer %>% 
  ingredients::ceteris_paribus(new_observation = df_subset, variables = "carat")

pd_carat = explainer %>% 
  ingredients::partial_dependency(variables = "carat")


plot(ices) +
  show_aggregated_profiles(pd_carat)


df_ices = ices %>% 
  as_tibble() %>% 
  rename(var_name = `_vname_`, y = `_yhat_`, id = `_ids_`) 


df_ices %>%
  ggplot(aes(carat, y)) +
  geom_line(aes(group = id), color = "gray", alpha = 0.5) +
  geom_line(aes(x, y), 
            data = filter(df_pd_num, var_name == "carat"),
            color = cols[1], size = 1.5) +
  scale_x_continuous(breaks = breaks_width(1)) + 
  labs(x = "carat", y = "log_price") + 
  theme_scatter()


save_plot("figure/pd_ices_carat.png")
# Conditional PD ----------------------------------------------------------


df_subset = df %>% 
  sample_n(100) %>%
  as.data.frame()

ices = explainer %>% 
  ingredients::ceteris_paribus(new_observation = df_subset, variables = "carat")

cpd = ices %>% 
  aggregate_profiles(groups = "clarity")

df_ices = ices %>% 
  as_tibble() %>% 
  rename(var_name = `_vname_`,  x = `_x_`, y = `_yhat_`, groups = `_groups_`) 

df_pred = model_trained %>% 
  predict(df_instance) %>% 
  bind_cols(df_instance)

df_cpd %>%
  ggplot(aes(x, y)) +
  geom_line(aes(group = groups), size = 1, color = cols[1]) +
  # geom_point(aes(carat, .pred), data = df_pred, 
  #            size = 4, shape = 21, color = "white", fill = cols[2]) + 
  scale_x_continuous(breaks = breaks_width(1)) + 
  labs(x = "carat", y = "log_price") + 
  theme_scatter()


save_plot("figure/ice_carat.png")


# SHapley Additive exPlanations -------------------------------------------

shap = explainer %>% 
  iBreakDown::shap(new_observation = df_instance, B = 5)

plot(shap)

mean_pred = model_trained %>% 
  predict(df) %>% 
  summarise(mean(.pred)) %>% 
  pull()

df_pred$.pred - mean_pred

df_shap = shap %>% 
  as_tibble() %>% 
  group_by(variable) %>% 
  summarise(contribution = mean(contribution)) %>% 
  mutate(variable = fct_reorder(variable, abs(contribution))) %>% 
  mutate(sign = if_else(contribution > 0, "positive", "negative"))


df_shap %>% 
  summarise(sum(contribution))

df_shap %>% 
  ggplot(aes(variable, contribution, fill = sign)) +
  geom_col(width = 0.8, alpha = 0.8) +
  coord_flip() + 
  scale_fill_manual(name = "", values = cols[c(2, 1)]) + 
  scale_y_continuous(breaks = breaks_width(0.2)) + 
  labs(x = NULL, y = "SHAP Value") + 
  theme_bar()

save_plot("figure/shap.png")





















