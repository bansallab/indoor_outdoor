# This file generates *smoothed* county-week estimates of the indoor/outdoor ratio
# from the raw *unsmoothed* estimates.

###########
# Libraries

library(tidyverse)
library(lubridate)
library(mgcv)
library(broom)
library(broom.mixed)
library(purrr)

############
# Read in raw estimates


df.r <- read_csv('data/indoor_outdoor_ratio_unsmoothed.csv')


df.r.raw.smooth <- df.r %>% 
  filter(!is.na(r_raw)) %>% 
  mutate(date_num = as.numeric(factor(week))) %>% 
  nest(data = -fips) %>% 
  mutate(test = map(data, ~ gam(.x$r_raw ~ 1 + s(.x$date_num))),
         pred = map(test, ~predict(.x, se = T))) %>% 
  unnest_wider(c(pred)) %>% 
  unnest(data, fit, se.fit) %>% 
  select(fips, week, fit, se.fit) %>% 
  mutate(type = 'raw')

df.r.weight.smooth <- df.r %>% 
  filter(!is.na(r_weighted_dwell)) %>% 
  mutate(date_num = as.numeric(factor(week))) %>% 
  nest(data = -fips) %>% 
  mutate(test = map(data, ~ gam(.x$r_weighted_dwell ~ 1 + s(.x$date_num))),
         pred = map(test, ~predict(.x, se = T))) %>% 
  unnest_wider(c(pred)) %>% 
  unnest(data, fit, se.fit) %>% 
  select(fips, week, fit, se.fit) %>% 
  mutate(type = 'weighted')


df.full <- full_join(df.r.raw.smooth, df.r.weight.smooth)

############
# Save smoothed estimates

write_csv(df.full, 'data/indoor_outdoor_ratio_smoothed.csv')

###########
