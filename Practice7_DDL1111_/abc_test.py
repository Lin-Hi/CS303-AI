import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_NQueens, plot_evolution

random.seed(time.time())

# now refactor things into a *Problem* abstraction
# you can directly reuse what you have implemented above
from abc import ABC, abstractmethod


class GAProblem(ABC):
    @abstractmethod
    def init_population(self, pop_size): pass

    @abstractmethod
    def fitness(self, sample): pass

    @abstractmethod
    def reproduce(self, population): pass

    @abstractmethod
    def replacement(self, old, new): pass


class PhraseGeneration(GAProblem):
    def __init__(self, target, alphabet):
        self.target = target
        self.alphabet = alphabet
        self.mutation_rate = mutation_rate

    def init_population(self, pop_size):
        # raise NotImplementedError()
        population = []
        for _ in range(pop_size):
            new_individual = "".join(random.choices(self.alphabet, k=len(self.target)))
            population.append(new_individual)

        return population

    def recombine(self, x, y):
        """
        TODO: combine two parents to produce an offspring
        """
        s = ''
        for i in range(len(x)):
            if x[i] == self.target[i]:
                s += x[i]
            elif y[i] == self.target[i]:
                s += y[i]
            else:
                if abs(ord(x[i]) - ord(self.target[i])) < abs(ord(y[i]) - ord(self.target[i])):
                    s += x[i]
                else:
                    s += y[i]
        # for i in range(len(x)):
        #     ran = random.random()
        #     if ran > 0.97:
        #         s = s[:i] + str(random.randint(0, 9)) + s[i + 1:]
        return s

    def select(self, r, population):
        """
        TODO: select *r* samples from *population*
        the simplest choice is to sample from *population* with each individual weighted by its fitness
        """
        population.sort(key=lambda x: self.fitness(x), reverse=True)
        bad_num = r // 3
        return population[:r - bad_num] + population[len(population) - bad_num:]

    def mutate(self, x, gene_pool, pmut):
        """
        apply mutation to *x* by randomly replacing one of its gene from *gene_pool*
        """
        ans = x
        if random.uniform(0, 1) >= pmut:
            return x
        l = [i for i in range(len(x))]
        for _ in range(1):
            i = l[random.randint(0, len(l) - 1)]
            l.remove(i)
            j = random.randrange(0, 10)
            ans = x[:i] + str(j) + x[i + 1:]
        for _ in range(1):
            n = len(x)
            g = len(gene_pool)
            c = random.randrange(0, n)
            r = random.randrange(0, g)
            new_gene = gene_pool[r]
            ans = x[:c] + new_gene + x[c + 1:]
        return ans

    def fitness(self, sample):
        # TODO: evaluate how close *sample* is to the target
        grades = 0
        for i in range(0, len(target)):
            if sample[i] == target[i]:
                if target[i].isdigit():
                    grades += 1
                else:
                    grades += 1
        return pow(grades,2)

    def reproduce(self, population, mutation_rate):
        """
        TODO: generate the next generation of population

        hint: make a new individual with

        mutate(recombine(*select(2, population, fitness_fn)), gene_pool, pmut)

        """
        ans = []
        size = len(population)
        s = set()
        best = max(population, key=self.fitness)
        for _ in range(len(population)):
            i, j = random.randrange(0, len(population) - 1), random.randrange(0, len(population) - 1)
            ans.append(self.mutate(self.recombine(best, population[j]), self.alphabet, mutation_rate))
        return ans

    def replacement(self, old, new):
        """
        You can use your own strategy, for example retain some solutions from the old population
        """
        l = old + new
        l.sort(key=lambda s: self.fitness(s), reverse=True)
        return l[:len(old)]


def genetic_algorithm(
        problem: GAProblem,
        ngen, n_init_size, mutation_rate,
        log_intervel=100
):
    population = problem.init_population(n_init_size)
    best = max(population, key=problem.fitness)
    history = [(0, list(map(problem.fitness, population)))]

    for gen in range(ngen):
        next_gen = problem.reproduce(population, mutation_rate)
        population = problem.replacement(population, next_gen)

        current_best = max(population, key=problem.fitness)

        if gen % log_intervel == 0:
            current_best = max(population, key=problem.fitness)
            if problem.fitness(current_best) > problem.fitness(best): best = current_best
            print(f"Generation: {gen}/{ngen},\tBest: {best},\tFitness={problem.fitness(best)}")
            history.append((gen, list(map(problem.fitness, population))))

    history.append((ngen - 1, list(map(problem.fitness, population))))
    return best, history


if __name__ == '__main__':
    # now set up the parameters
    ngen = 200
    max_population = 150
    mutation_rate = 0.5

    sid = 12010903  # TODO:  replace this with your own sid
    target = f"Genetic Algorithm by {sid}"
    u_case = [chr(x) for x in range(65, 91)]
    l_case = [chr(x) for x in range(97, 123)]
    num_case = [str(i) for i in range(10)]
    alphabet = u_case + l_case + num_case + [" "]  # TODO: fix this: what is the search space now?

    problem = PhraseGeneration(target, alphabet)

    # and run it
    solution, history = genetic_algorithm(problem, ngen, max_population, mutation_rate,5)
