#Kacper Tumulec 44535
#SI LAB8

import random
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

class evolution(object):
    def __init__(self, n=20, pop=20, genmax=100, pc=0.7, pm=0.2):
        self.n = n
        self.pop = pop
        self.genmax = genmax
        self.pc = pc
        self.pm = pm

    def _generateyounglings(self):
        listofyounglings = []
        for i in range(self.pop):
            n = list(range(1, self.n+1))
            random.shuffle(n)
            listofyounglings.append(n)
        return listofyounglings

    def _evaluate(self, entrylist):
        hitcount = np.zeros(self.pop)
        for iter, element in enumerate(entrylist):
            for i in range(len(element)):
                posx_up = i+1
                posy_up = element[i]

                posx_down = i+1
                posy_down = element[i]

                # go upper right
                if posy_up != 1:
                    while posy_up > 1 and posx_up < self.n:
                        posx_up = posx_up+1
                        posy_up = posy_up-1
                        if(element[posx_up-1] == posy_up):
                            hitcount[iter] = hitcount[iter] + 1

                # go lower right
                if posy_down != self.n:
                    while posy_down < self.n and posx_down < self.n:
                        posx_down = posx_down+1
                        posy_down = posy_down+1
                        if(element[posx_down-1] == posy_down):
                            hitcount[iter] = hitcount[iter] + 1
        self.curr_evaluation = hitcount

    def _selection(self, P):
        i = 0
        resultpop = []
        while i < self.pop:
            o1, o2 = random.sample(range(self.pop), 2)
            if not np.array_equal(P[o1], P[o2]):
                if self.curr_evaluation[o1] <= self.curr_evaluation[o2]:
                    tempsele = P[o1][:]
                    resultpop.append(tempsele)
                else:
                    tempsele = P[o2][:]
                    resultpop.append(tempsele)
                i+=1
        return resultpop

    def _crossover(self, Pn):
        i = 0
        while i <= self.pop-2:
            if random.uniform(0, 1) <= self.pc:
                self._cross(Pn[i], Pn[i+1])
            i+=2
        return Pn

    def _cross(self, o1, o2):
        csize = random.randint(1, self.n-1)
        cstart = random.randint(0, self.n-csize)
        mappinglist1 = []
        mappinglist2 = []
        for i in range(csize):
            tempval = o1[cstart+i]
            mappinglist1.append(tempval)
            o1[cstart+i] = o2[cstart+i]
            mappinglist2.append(o2[cstart+i])
            o2[cstart+i] = tempval
        while len(o1) > len(set(o1)) or len(o2) > len(set(o2)):
            j = 0
            while j < self.n:
                if j == cstart:
                    j += csize
                    continue
                else:
                    if o1[j] in mappinglist2:
                        if mappinglist1[mappinglist2.index(o1[j])] in o1:
                            o1[j] = mappinglist1[mappinglist2.index(mappinglist1[mappinglist2.index(o1[j])])]
                        else:
                            o1[j] = mappinglist1[mappinglist2.index(o1[j])]
                    if o2[j] in mappinglist1:
                        if mappinglist2[mappinglist1.index(o2[j])] in o2:
                            o2[j] = mappinglist2[mappinglist1.index(mappinglist2[mappinglist1.index(o2[j])])]
                        else:
                            o2[j] = mappinglist2[mappinglist1.index(o2[j])]
                j += 1

    def _mutation(self, Pn):
        i = 0
        while i < self.pop:
            if random.uniform(0, 1) < self.pm:
                self._mutate(Pn[i])
            i += 1
        return Pn

    def _mutate(self, o1):
        idx1, idx2 = random.sample(range(self.n), 2)
        temp = o1[idx1]
        o1[idx1] = o1[idx2]
        o1[idx2] = temp


    def EVOLVE(self, plot = True):
        best_scores = []
        avg_scores = []
        generation_list = []
        younglings = self._generateyounglings()
        self._evaluate(younglings)
        gen = 0
        best = np.argmin(self.curr_evaluation)
        best_scores.append(self.curr_evaluation[best])
        avg_scores.append(mean(self.curr_evaluation))
        generation_list.append(gen)
        while gen < self.genmax and self.curr_evaluation[best] > 0:
            younglings = self._selection(younglings)
            younglings = self._crossover(younglings)
            younglings = self._mutation(younglings)
            self._evaluate(younglings)
            best = np.argmin(self.curr_evaluation)
            gen = gen + 1
            best_scores.append(self.curr_evaluation[best])
            avg_scores.append(mean(self.curr_evaluation))
            generation_list.append(gen)
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(generation_list, best_scores)
            ax2.plot(generation_list, avg_scores)
            ax1.set(xlabel ='Generation', ylabel='Fitness function (best)')
            ax2.set(xlabel ='Generation', ylabel='Fitness function (mean)')
            plt.show()
        return younglings[int(best)], int(self.curr_evaluation[best])






def main():
    pokemon = evolution()
    print(pokemon.EVOLVE())



if __name__ == "__main__":
    main()

