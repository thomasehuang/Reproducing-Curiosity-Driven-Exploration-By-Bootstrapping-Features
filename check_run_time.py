#!/bin/sh python3

S_IN_M = 60
M_IN_H = 60
S_IN_H = S_IN_M*M_IN_H
H_IN_DAY = 24
S_IN_DAY = S_IN_H*H_IN_DAY


hours_avail = 10
n_eval = 1000
s_per_n_eval = 15
evals_per_s = 1.0*n_eval/s_per_n_eval
evals_per_hour = evals_per_s*S_IN_H
print(round(evals_per_hour*hours_avail), "evals in", hours_avail, "hours")


desired_n_eval = 10e6
s_to_desired_n_eval = desired_n_eval/evals_per_s
print(s_to_desired_n_eval/S_IN_DAY, "days")

budget = 30.0
price_per_hour = 0.30
print(budget/price_per_hour, "hours of credit remaining")
print(budget/price_per_hour/H_IN_DAY, "days of credit remaining")
