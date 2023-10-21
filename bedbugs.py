import numpy as np
import matplotlib.pyplot as plt

#Number of bed bugs that dies, strays, etc.
mortality_rate = 0.03
#Number of pregnant female bed bugs at day 0
F0 = 1
#Number of newborn nymphs at day 0
N0 = 0
#Number of days after infestation
limit = 80
#Days until bed bugs become mature to breed
maturity = 37
#Number of eggs a mature female bed bug lays per day
birth_rate = 1

#Total number of adult females over time
F=[0 for x in range(limit)]
F[0] = F0
#Number of births per day
B=[0 for x in range(limit)]
B[0] = N0
#Total number of males and nymph (non-egglaying bugs) over time
N=[0 for x in range(limit)]
N[0] = N0

def death_propability(n,p):
    deaths = 0
    for i in range(n):
        if np.random.rand() < mortality_rate:
            deaths += 1
    return deaths

def new_bugs(F,B,N):
    for t in range(1,limit):
        if t>maturity:
            F[t] = F[t-1] + birth_rate*(B[t-maturity-1])/2
            B[t] = birth_rate*F[t] - death_propability(int(birth_rate*F[t]),mortality_rate)
            N[t] = N[t-1] + B[t] - B[t-maturity-1]/2
        else:
            F[t] = F[t-1]
            B[t] = birth_rate*F[t] - death_propability(int(birth_rate*F[t]),mortality_rate)
            N[t] = N[t-1] + B[t]

    return F,B,N

def visualize(y):
    x = np.linspace(1,limit,limit)
    plt.plot(x,y)
    plt.show()

F,B,N = new_bugs(F,B,N)
total = [a+b for a, b in zip(F, N)]