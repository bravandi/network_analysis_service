import random
import bisect
import collections

def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]

weights=[0.3, 0.4, 0.3]
population = 'ABC'
counts = collections.defaultdict(int)
for i in range(10000):
    counts[choice(population, weights)] += 1
print(counts)

# % test.py
# defaultdict(<type 'int'>, {'A': 3066, 'C': 2964, 'B': 3970})
