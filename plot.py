#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 07:15:40 2019

@author: dennis
"""

import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    with open("results_DAgger_long.pkl", "rb") as f:
        dagger = pickle.load(f, encoding='latin1')
        
    print(dagger)
    
    x = range(len(dagger['train_size']))
    
    fig, ax = plt.subplots()
    
    plt.title("DAgger versus NoDAgger performance for Optimal Control")
    plt.xlabel("Iteration [10 rollouts per iteration]")
    plt.ylabel("Mean reward")
    plt.ylim((0,11000))
    y = [dagger['expert_mean']] * len(x)
    yerr = [dagger['expert_std']] * len(x)
    
    
    plt.plot(x,y, 'o-', label="Expert Performance")
    plt.plot(x, dagger['means'], 'o-', label="DAgger")
    
    with open("results_NoDAgger_mixed.pkl", "rb") as f:
        nodagger = pickle.load(f, encoding="latin1")
    
    plt.plot(x, nodagger['means'],'o-', label="NoDAgger (mixed, p_0=0.9)")
    
    with open("results_NoDAgger.pkl", "rb") as f:
        nodagger2 = pickle.load(f, encoding="latin1")
    plt.plot(x, nodagger2['means'],'o-', label="NoDAgger")
    
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("control.png", bbox_extra_artists=(lgd,), bbox_inches='tight')