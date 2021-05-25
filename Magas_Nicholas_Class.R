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

loan_test <- read_csv("data/test.csv")



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





# Examining boosted tree performance

load(file = "data/bt_tune.rda")


bt_workflow_tuned <- bt_workflow %>% 
  finalize_workflow(select_best(bt_tune, metric = "accuracy"))


bt_results <- fit(bt_workflow_tuned, loan_train)


metrics <- metric_set(accuracy)



predict(bt_results, new_data = loan_test) %>%
  bind_cols(loan_test %>% select(hi_int_prncp_pd)) %>%
  bind_cols(predict(bt_results, new_data = loan_test, type = "prob")) %>%
  metrics(truth = hi_int_prncp_pd, estimate = .pred_class, .pred_yes)















# submission code

rf_final_predictions <- predict(rf_results, new_data = loan_test)



submit <- read_csv("data/sample submission.csv") %>% 
  bind_cols(rf_final_predictions) %>% 
  select(-Category) %>% 
  rename(Category = .pred_class)



write_csv(file = "class_results.csv", submit)





