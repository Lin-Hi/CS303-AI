from abc import ABC, abstractmethod
import random


class GAProblem(ABC):
    @abstractmethod
    def init_population(self, pop_size): pass

    @abstractmethod
    def fitness(self, sample): pass

    @abstractmethod
    def reproduce(self, population): pass

    @abstractmethod
    def replacement(self, old, new): pass


class NQueensProblem(GAProblem):
    def __init__(self, n):
        self.n = n
        self.max_fitness = n * (n - 1) // 2  # max number if non-attacking pairs

    def init_population(self, pop_size):
        population = []
        for _ in range(pop_size):
            l = [str(i) for i in range(self.n)]
            init = ''
            while len(l) != 0:
                i = random.choice(l)
                init += i
                l.remove(i)
            population.append(init)
        return population

    def fitness(self, queens):
        value = 28
        for i in range(self.n):
            for j in range(i + 1,self.n):
                if queens[i] == queens[j] or abs(i - j) == abs(int(queens[i]) - int(queens[j])):
                    value -= 1
        return value

    def selcct(self, population, r):
        selected = []
        not_chosen = population.copy()
        for _ in range(r):
            chosen = random.choice(not_chosen)
            selected.append(chosen)
            not_chosen.remove(chosen)
        return selected

    def reproduce(self, population, mutation_rate):
        # TODO:alomost the same as the previous problem.
        next_population = []
        for _ in range(len(population) // 2):
            f, m = self.selcct(population, 2)
            r = random.randint(0, len(population) - 1)
            c1 = f[:r] + m[r:]
            c2 = m[:r] + f[r:]

            not_in = [str(i) for i in range(self.n)]
            count = [0 for _ in range(self.n)]
            for i in range(self.n):
                if c1[i] in not_in:
                    not_in.remove(c1[i])
                count[int(c1[i])] += 1
            for i in range(len(count)):
                if count[i] > 1:
                    c1.replace(str(i), not_in[0], 1)
                    del not_in[0]

            not_in = [str(i) for i in range(self.n)]
            count = [0 for _ in range(self.n)]
            for i in range(self.n):
                if c2[i] in not_in:
                    not_in.remove(c2[i])
                count[int(c2[i])] += 1
            for i in range(len(count)):
                if count[i] > 1:
                    c2.replace(str(i), not_in[0], 1)
                    del not_in[0]
            next_population.append(self.mutate(c1, mutation_rate))
            next_population.append(self.mutate(c2, mutation_rate))
        return next_population

    def mutate(self, queens, mutation_rate):
        r = random.random()
        if r > mutation_rate:
            return queens

        l = [i for i in range(len(queens))]
        i = int(random.choice(l))
        l.remove(i)
        j = random.choice(l)
        l.remove(j)
        i_, j_ = queens[i], queens[j]
        queens = queens.replace(queens[i], 'x')
        queens = queens.replace(queens[j], i_)
        queens = queens.replace('x', j_)
        return queens

    def replacement(self, old, new):
        """
        You can use your own strategy, for example retain some solutions from the old population
        """
        all = old + new
        all.sort(key=lambda x: self.fitness(x), reverse=True)
        return all[:len(old)]

    def __repr__(self):
        return f"{self.n}-Queens Problem"


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

        if gen % log_intervel == 0:
            current_best = max(population, key=problem.fitness)
            if problem.fitness(current_best) > problem.fitness(best): best = current_best
            print(f"Generation: {gen}/{ngen},\tBest: {best},\tFitness={problem.fitness(best)}")
            history.append((gen, list(map(problem.fitness, population))))

    history.append((ngen - 1, list(map(problem.fitness, population))))
    return best, history


if __name__ == '__main__':
    from utils import plot_NQueens

    ngen = 1000
    init_size = 100
    mutation_rate = 0.5

    n = 8
    problem = NQueensProblem(n)
    solution, history = genetic_algorithm(problem, ngen, init_size, mutation_rate,100)
