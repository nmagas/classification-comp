# Model Setup, Tuning, and Submission for Random Forest model all included in this script
# GitHub link is https://github.com/nmagas/classification-comp.git

# Loading Packages----
library(tidyverse)
library(tidymodels)
library(naniar)
library(skimr)


# setting seed
set.seed(42)


# reading in data
loan <- read_csv("data/train.csv") %>% 
  mutate(hi_int_prncp_pd = factor(hi_int_prncp_pd))

final_loan_test <- read_csv("data/test.csv")



# checking for missingness
miss_var_summary(loan)

skim_without_charts(loan)




# splitting data
loan_split <- initial_split(loan, prop = 0.7, strata = hi_int_prncp_pd)
loan_train <- training(loan_split)
loan_test <- testing(loan_split)




# creating folds
loan_folds <- vfold_cv(loan_train, v = 10, repeats = 5, strata = hi_int_prncp_pd)




# creating recipe
loan_recipe <- recipe(hi_int_prncp_pd ~ ., data = loan_train) %>% 
  step_rm(id, addr_state, earliest_cr_line, emp_length, emp_title, last_credit_pull_d, 
          purpose, sub_grade, home_ownership) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_normalize(all_predictors()) %>% 
  step_zv(all_predictors())







# prepping and baking recipe

loan_recipe %>% 
  prep(loan_train) %>% 
  bake(new_data = NULL)





# saving objects
save(loan_folds, file = "data/loan_folds.rda")
save(loan_recipe, file = "data/loan_recipe.rda")



# Random Forest Tuning----


# defining model
rf_model <- rand_forest(mtry = tune(), min_n = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")



# checking parameters
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(1, 10)))



# defining tuning grid
rf_grid <- grid_regular(rf_params, levels = 5)


# creating workflow
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(loan_recipe)


# tuning
rf_tune <- rf_workflow %>% 
  tune_grid(resamples = loan_folds, grid = rf_grid)









# writing out results and workflow

save(rf_tune, rf_workflow, file = "data/rf_tune.rda")






# Examining random forest performance

load(file = "data/rf_tune.rda")


rf_workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tune, metric = "accuracy"))


rf_results <- fit(rf_workflow_tuned, loan_train)


metrics <- metric_set(accuracy)



predict(rf_results, new_data = loan_test) %>%
  bind_cols(loan_test %>% select(hi_int_prncp_pd)) %>%
  bind_cols(predict(rf_results, new_data = loan_test, type = "prob")) %>%
  metrics(truth = hi_int_prncp_pd, estimate = .pred_class, .pred_yes)







# submission code for rf model

rf_final_predictions <- predict(rf_results, new_data = final_loan_test)



submit <- read_csv("data/sample submission.csv") %>% 
  bind_cols(rf_final_predictions) %>% 
  select(-Category) %>% 
  rename(Category = .pred_class)


write_csv(file = "class_results3.csv", submit)









