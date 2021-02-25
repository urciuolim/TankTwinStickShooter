import math

def elo_prob(elo1, elo2):
    return 1. / (1. + math.pow(10., (elo2-elo1)/400.))

def elo_change(elo_a, elo_b, K, a_win_rate):
    a_prob = elo_prob(elo_a, elo_b)
    b_prob = elo_prob(elo_b, elo_a)
    
    a_elo_change = K * (a_win_rate - a_prob)
    b_elo_change = K * ((1-a_win_rate) - b_prob)
    
    return round(a_elo_change), round(b_elo_change)