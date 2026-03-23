import probility_distribution as m
print('bitlist->dec', m.bitlist_to_decimal([1,0,1,1]))
dqnn_samples, isqnn_samples, dqnn_hist, isqnn_hist, bins = m.generate_probability_distribution('000011110000', 3, 4, [0.1]*12, num_samples=5, use_cirq=False)
print('ok', dqnn_samples, dqnn_hist)
