# Boosted Tree Tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(xgboost)

# load required objects ----
load(file = "data/loan_folds.rda")
load(file = "data/loan_recipe.rda")


bt_loan_recipe <- recipe(hi_int_prncp_pd ~ ., data = loan_train) %>% 
  step_rm(id, addr_state, earliest_cr_line, emp_length, emp_title, last_credit_pull_d, 
          purpose, sub_grade, home_ownership) %>% 
  step_dummy(all_nominal(), -hi_int_prncp_pd, one_hot = TRUE) %>% 
  step_normalize(all_predictors()) %>% 
  step_zv(all_predictors())



# Define model ----
bt_model <- boost_tree(mtry = tune(), min_n = tune(), learn_rate = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("xgboost")






# set-up tuning grid ----

# checking parameters
bt_params <- parameters(bt_model) %>% 
  update(mtry = mtry(range = c(1, 10)),
         learn_rate = learn_rate(range = c(-5, -0.2)))



# define tuning grid
bt_grid <- grid_regular(bt_params, levels = 5)


# workflow ----
bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(bt_loan_recipe)


# Tuning/fitting ----

# Place tuning code in here
bt_tune <- bt_workflow %>% 
  tune_grid(resamples = loan_folds, 
            grid = bt_grid)




# Write out results & workflow

save(bt_tune, bt_workflow, file = "data/bt_tune.rda")
