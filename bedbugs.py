import numpy as np
import matplotlib.pyplot as plt

def prob(n,p):
    #Checks the chance of an event happening to n bed bud bugs with probability p
    bugs = 0
    for i in range(n):
        if np.random.rand() < p:
            bugs += 1
    return bugs

def visualize(y, legend,start, stop, scale='linear'):
        x = np.linspace(start,stop,stop-start)
        y = y[start:stop]
        plt.plot(x,y,label=legend)

def exp_fit(T):
    x = np.linspace(0,d,d)
    fit = np.polyfit(x, np.log(T), 1)

    y = np.exp(fit[1])*np.exp(fit[0]*x)
    
    return y

class BedBugs:
    def __init__(self,
                max_days,
                initial_bugs = [0,0,0],
                maturity = 37,
                hatching = 8,
                birth_rate = 1,
                severity = 1,
                mortality_rate= 0.05,
                blood_per_bite = 0.055,
                feeding_frequency= 5,
                randomized = True,
                mating = True):
        
        self.max_days = max_days #Number of days after initial infestation
        self.F0 = initial_bugs[0] #Number of adult female bed bugs at day 0
        self.M0 = initial_bugs[1] #Number of adult male bed bugs at day 0
        self.N0 = initial_bugs[2] #Number of newlaid eggs at day 0
        self.maturity = maturity #Number of days until bed bugs become mature to breed
        self.hatching = hatching #Days until eggs hatch
        self.birth_rate = birth_rate #Number of eggs a adult female bed bug lays per day
        self.severity = severity #Determines the severity of the infestation
        self.mortality_rate = mortality_rate #Chance that a bed bug doesn't survive to become an adult
        self.blood_per_bite = blood_per_bite #The average amount of blood consumed per bite in milliliters
        self.feeding_frequency = feeding_frequency #Number of days between blood meals for a single bed bug
        self.randomized = randomized #Randomizes the initial conditions of the infestation
        self.mating = mating #Decides wether a female needs an adult male to mate with in order to lay eggs
        
        self.B = [0 for x in range(maturity+max_days)] #Number of new births at given day
        self.E = [0 for x in range(max_days)] #Total number of eggs over time
        self.N = [0 for x in range(max_days)] #Total number of eggs and nymphs (non-adult bugs) over time
        self.F = [0 for x in range(max_days)] #Total number of adult females (egglaying bugs) over time
        self.M = [0 for x in range(max_days)] #Total number of adult males over time
        self.T = [0 for x in range(max_days)] #Total number of bed bugs (adult males, adult females, nymphs) in an infestation
        self.BL = [0 for x in range(max_days)] #The amount of blood lost per day from bed bug bites
    
    def initial_bug_randomizer(self): #initializes a bed bug infestation with bugs of various ages and numbers
        self.B[0] = np.random.binomial(self.severity,0.8) #Since adult bugs live longer than nymphs, there's a much higher change that one of the initial bugs are an adult
        for i in range(1,self.maturity-self.hatching):
            self.B[i] = np.random.binomial(self.severity,0.05)

    def bed_bug_growth(self): #Calculates the growth of bed bugs over time
        #Initializes eggs and nymphs at day 0
        self.E[0] = sum(self.B[self.maturity-self.hatching:self.maturity])
        self.N[0] = self.N0 + sum(self.B[1:self.maturity-self.hatching])

        #Initializes adult bed bugs at day 0
        gender_selection = prob(self.B[0],0.5)
        self.F[0] = self.F0 + gender_selection
        self.M[0] = self.M0 + (self.B[0]-gender_selection)
        
        for t in range(1,self.max_days):
            t_B = t+self.maturity
            #Selects the gender of nymphs turning into adults
            gender_selection = prob(self.B[t],0.5)
            self.F[t] = self.F[t-1] + gender_selection
            self.M[t] = self.M[t-1] + (self.B[t]-gender_selection)

            if self.mating==False or (self.M[t-1] != 0 and self.F[t-1] != 0): #Checks if there's any egg-laying females present
                daily_birth_rate = int(self.birth_rate*self.F[t-1]) #Number of eggs layed per day
                mortality = prob(daily_birth_rate,self.mortality_rate) #Calculates the chance of newborn dying
                self.B[t_B] = daily_birth_rate + mortality

            hatching_bugs = self.B[t_B-self.hatching]
            self.E[t] = self.E[t-1] + self.B[t_B] - hatching_bugs
            self.N[t] = self.N[t-1] - self.B[t] + hatching_bugs
    
    def initializer(self): #Initializes a bed bug infestation        
        if self.randomized:
            self.initial_bug_randomizer()
        self.bed_bug_growth()
        self.H = [a+b+c for a,b,c in zip(self.F,self.N,self.M)]
        self.T = [a+b for a,b in zip(self.H,self.E)]

        self.init = self.B[:self.maturity]
        self.births = self.B[self.maturity:]

    def blood_loss(self): #Calculates blood loss per day from bed bug bites
        for i in range(len(self.BL)):
            self.blood_per_day[i] = (self.T[i]//self.feeding_frequency)*self.blood_per_bite