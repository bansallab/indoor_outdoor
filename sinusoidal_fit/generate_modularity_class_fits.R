# This file generates the fitted  curves to each of the four identified clusters

##################
# Libraries

library(tidyverse)
library(lubridate)
library(lme4)
library(nlme)
library(broom.mixed)
library(mgcv)

##################s
# Read in data

clusters <- read_csv('data/fips_modulclass.csv') %>% 
  select(-X1) %>% 
  rename(fips = node) %>% 
  mutate(fips = as.double(fips))

df.fips <- read_csv('data/state_and_county_fips_master.csv') %>% 
  mutate(fips = if_else(fips ==02270, 02158, fips),
         fips = if_else(fips == 46113, 46102, fips))

df.full <- read_csv('data/indoor_outdoor_ratio_unsmoothed.csv') %>% 
  left_join(df.fips) %>% 
  left_join(clusters) %>% 
  filter(!is.na(fips), !is.na(state), !is.na(week), !is.na(modularity_class)) %>% 
  group_by(fips) %>% 
  arrange(week) %>%
  mutate(t = row_number()) %>% 
  ungroup() %>% 
  mutate(state = as.factor(state))

###########
# Fit sine curve estimates 

params <- df.full %>% 
  nest(data = -modularity_class) %>% 
  mutate(fit = map(data, ~ nls(r_raw ~ A*sin(omega*t+phi)+C, 
                               data=.x, 
                               start=c(A=.25,omega=.127,phi=1,C=.98))),
         tidied = map(fit, tidy),
         preds = map(fit, predict, newdata = tibble(t=1:182)))

###########
# Save sine curve fits

params %>% 
  select(modularity_class, tidied) %>% 
  unnest(tidied) %>% 
  write_csv('data/sine_curve_cluster_fits.csv')

params %>% 
  select(modularity_class, preds) %>% 
  unnest(preds) %>% 
  group_by(modularity_class) %>% 
  mutate(t = row_number()) %>% 
  ungroup() %>% 
  left_join(df.full %>% select(week, t) %>% unique()) %>% 
  write_csv('data/sine_curve_cluster_preds.csv')

#########
# Fit GAM

params <- df.full %>% 
  nest(data = -modularity_class) %>% 
  mutate(fit = map(data, ~ gam(r_raw ~ s(t), 
                               data=.x)),
         tidied = map(fit, tidy),
         preds = map(fit, predict, newdata = tibble(t=1:182)))

#########
# Save GAM estimates

params %>% 
  select(modularity_class, tidied) %>% 
  unnest(tidied) %>% 
  write_csv('data/gam_cluster_fits.csv')

params %>% 
  select(modularity_class, preds) %>% 
  unnest(preds) %>% 
  group_by(modularity_class) %>% 
  mutate(t = row_number()) %>% 
  ungroup() %>% 
  left_join(df.full %>% select(week, t) %>% unique()) %>% 
  write_csv('data/gam_cluster_preds.csv')
