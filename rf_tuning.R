# Random Forest Tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)

set.seed(42)

# load required objects ----
load(file = "data/loan_folds.rda")
load(file = "data/loan_recipe.rda")

# Define model ----
rf_model <- rand_forest(mtry = tune(), min_n = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")



# set-up tuning grid ----

# checking parameters
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(1, 10)))



# define tuning grid
rf_grid <- grid_regular(rf_params, levels = 5)


# workflow ----
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(loan_recipe)


# Tuning/fitting ----

# Place tuning code in here
rf_tune <- rf_workflow %>% 
  tune_grid(resamples = loan_folds, grid = rf_grid)




# Write out results & workflow

save(rf_tune, rf_workflow, file = "data/rf_tune.rda")
