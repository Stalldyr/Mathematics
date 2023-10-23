import numpy as np
import matplotlib.pyplot as plt

mortality_rate = 0.05 #Number of bed bugs that dies, strays, etc.
F0 = 1 #Number of pregnant female bed bugs at day 0
N0 = 0 #Number of newborn nymphs at day 0
limit = 60 #Number of days after initial infestation
maturity = 37 #Number of days until bed bugs become mature to breed
birth_rate = 1 #Number of eggs a adult female bed bug lays per day
blood_per_bite = 0.055 #The average amount of blood consumed per bed bug in milliliters
feeding_frequency = 5 #Number of days between blood meals for a single bed bug

#F = Total number of adult females (egglaying bugs) over time
F=[0 for x in range(limit)]
F[0] = F0
#B = Number of births at given day
B=[0 for x in range(limit)]
B[0] = N0
#N = Total number of males and nymphs (non-egglaying bugs) over time
N=[0 for x in range(limit)]
N[0] = N0

def mortality(n,p):
    #Adjustes for the case where a bug doesn't survive to become an adult 
    deaths = 0
    for i in range(n):
        if np.random.rand() < p:
            deaths += 1
    return deaths

def new_bugs(F,B,N):
    #Calculates the number of bed bugs (both female and non-female) and births over time
    for t in range(1,limit):
        #birth_rate = np.random.choice([0,1,2,3,4], p=[0.05, 0.65, 0.25, 0.035, 0.015])
        if t>maturity:
            F[t] = F[t-1] + birth_rate*(B[t-maturity-1])/2
            B[t] = birth_rate*F[t] - mortality(int(birth_rate*F[t]),mortality_rate)
            N[t] = N[t-1] + B[t] - B[t-maturity-1]/2
        else:
            F[t] = F[t-1]
            B[t] = birth_rate*F[t] - mortality(int(birth_rate*F[t]),mortality_rate)
            N[t] = N[t-1] + B[t]

    return F,B,N

def visualize(y,start=1,stop=limit,scale='linear'):
    x = np.linspace(start,stop,stop-start+1)
    y = y[start-1:stop]
    plt.plot(x,y)
    plt.yscale(scale)
    plt.show()

F,B,N = new_bugs(F,B,N)

#The total number of bed bugs (both egglaying and non-egglaying) in an infestation
total_bugs = [a+b for a, b in zip(F, N)]