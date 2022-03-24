import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
#random.seed(101)


class GeneExpressionProgramming():

    def __init__(self, nhead, func_set, term_set, const_range, operator_probabilities):

        self.gen_pop_fit_history = {}
        self.ngenerations = None
        self.nhead = nhead

        self.func_set = func_set
        self.term_set = term_set

        self.one_arity_func = ['(sqrt)', '(sin)', '(exp)', '(ln)', '(inv)', '(gau)', '(X2)']
        self.two_arity_func = ['+', '-', '*', '/']

        self.max_arity = 1
        for func in func_set:
            if func in self.two_arity_func:
                self.max_arity = 2
                break

        self.ntail = self.nhead * (self.max_arity - 1) + 1
        self.chrom_length = self.nhead + self.ntail

        self.dc_length = self.ntail
        self.const_list = np.random.uniform(const_range[0],const_range[1],self.dc_length)
        #self.const_list = [3.0,1.1,4.0,1.1,4.0,1.1,4.0,1.1]

        self.operator_probabilities = operator_probabilities


    def VisualizeResults(self):
        ##Visualize results: plot [averagefitness, best fitness] vs generation
        average_fitness = [self.gen_pop_fit_history[i]['Mean Fitness'] for i in range(self.ngenerations + 1)]
        max_fitness = [self.gen_pop_fit_history[i]['Max Fitness Value'] for i in range(self.ngenerations + 1)]

        generation = [i for i in range(self.ngenerations + 1)]
        plt.plot(generation, average_fitness, label='Avg Fitness', marker='.')
        plt.plot(generation, max_fitness, label='Max Fitness', marker='.')
        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value (Max 1000)')
        plt.title('Fitness Value vs Generation')
        plt.savefig('Fitness Value vs Generation.png')
        plt.show()

        average_fitness_mse = [self.gen_pop_fit_history[i]['Mean mse'] for i in range(self.ngenerations + 1)]
        max_fitness_mse = [self.gen_pop_fit_history[i]['Fittest mse'] for i in range(self.ngenerations + 1)]
        #plt.plot(generation, average_fitness_mse, label='Avg MSE', marker='.')
        plt.plot(generation, max_fitness_mse, label='Fittest MSE', marker='.')
        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value (Min 0)')
        plt.title('MSE vs Generation')
        plt.savefig('MSE vs Generation.png')
        plt.show()

    def RunGEP(self, x, y, popsize, ngenerations, fitness_func):

        def AddGenInfo(gen_pop_fit_history, generation, population, fitness, fitness_mse):

            '''Add an entry to the generation:{population,history} info dictionary
            from current generation, population ORF list and fitness list'''
            import statistics
            gen_pop_fit_history[generation] = {
                'Population': population, 'Fitness': fitness,
                'Max Fitness Value': max(fitness),
                'Fittest Chromosome': population[fitness.index(max(fitness))].copy(),
                'Mean Fitness': statistics.mean(fitness),
                'Fittest mse': min(fitness_mse),
                'Mean mse': statistics.mean(fitness_mse)
            }
            #'Chromosome,Fitness': set(zip(population, fitness)),
            #print(gen_pop_fit_history[generation])
            return gen_pop_fit_history

        def ChromToET(chromosome):
            '''Take a string of chromosome change it to
            a dictionary of expression tree{row:element_on_row}'''
            expr_tree = {0: [chromosome[0]]}

            i = 1
            start_counter = 1
            while True:

                take_next = 0
                terminal_row = True

                for element in expr_tree[i - 1]:
                    if element in self.two_arity_func:
                        terminal_row = False
                        take_next += 2
                    elif element in self.one_arity_func:
                        terminal_row = False
                        take_next += 1

                if terminal_row==True:
                    break

                last_counter = start_counter + take_next
                next_elem = chromosome[start_counter:last_counter]
                expr_tree[i] = next_elem
                start_counter = last_counter

                i += 1

            return expr_tree

        def EvaluateET(chromosome, variable_dict):
            '''Take chromosome and terminal variable dictionary{'symbol':value}
            and perform calculation from the chromosome->ET->calculation->prediction'''

            # Change string to list in each row of ET and change variables to sample value
            expr_tree = ChromToET(chromosome)
            el_dc = 0
            for i in range(len(expr_tree)): #iterate rows
                el = 0
                #el_dc = 0
                for element in expr_tree[i]: #iterate elements in a row
                    if element in variable_dict.keys():
                        expr_tree[i][el] = str(variable_dict[element])
                    elif element == '?':
                        expr_tree[i][el] = str(self.const_list[el_dc])
                        el_dc += 1

                    el += 1


            def operate_two_arity(representation, a, b):
                a = float(a)
                b = float(b)

                if representation=='+':
                    result = a + b
                elif representation=='-':
                    result = a - b
                elif representation=='*':
                    result = a * b
                elif representation=='/':
                    if b==0:
                        b = 1e-6
                    result = a / b

                return str(result)

            def operate_one_arity(representation, a):
                a = float(a)

                if representation=='(sqrt)':
                    if a>=0:
                        result = a**0.5
                    else:
                        result = (abs(a))**0.5
                    #result = math.sqrt(a)

                elif representation=='(sin)':
                    result = math.sin(a)
                elif representation=='(exp)':
                    try:
                        result = math.exp(a)
                    except:
                        result = 1e6
                elif representation=='(ln)':
                    if a==0:
                        a = 1e-6
                    elif a<0:
                        a = abs(a)
                    result = math.log(a,math.e)
                elif representation=='(inv)':
                    if a==0:
                        a = 1e-6
                    result = 1/a
                elif representation=='(gau)':
                    result = np.random.normal(1)
                elif representation=='(X2)':
                    if a>1e4:
                        result = 1e6
                    else:
                        result = a**2

                return str(result)


            for row in range(len(expr_tree) - 2, -1, -1): #iterate rows from second last row to root
                i = 0
                for element in expr_tree[row]:
                    if element in self.two_arity_func:
                        a = expr_tree[row + 1][0]
                        b = expr_tree[row + 1][1]

                        result = operate_two_arity(element, a, b)
                        # buang 2 elemen pertama di row+1 dan replace elemen pertama di row
                        expr_tree[row + 1] = expr_tree[row + 1][2:]
                        expr_tree[row][i] = result

                    elif element in self.one_arity_func:
                        a = expr_tree[row + 1][0]
                        result = operate_one_arity(element, a)
                        expr_tree[row + 1] = expr_tree[row + 1][1:]
                        expr_tree[row][i] = result

                    i += 1

            prediction = float(expr_tree[0][0])
            return prediction

        def EvalFitness(chromosome, x, y, fitness_func):
            '''Take the string of a single chromosome and change it to Expression Tree
            then perform prediction with the ET and calculate fitness from
            the prediction against the groundtruth data or label'''
            ##Take a string of chrom,change from string to executable ET then evaluate its fitness from dataset
            if fitness_func == 'mse':
                squared_error_list = []
                for i in range(len(pd.DataFrame(x))):
                    variable_dict = {}
                    nth_input = 0
                    for term in self.term_set:
                        if term != '?':
                            variable_dict[term] = pd.DataFrame(x).iloc[i, nth_input]
                            nth_input += 1

                    prediction = EvaluateET(chromosome, variable_dict)
                    # print(prediction)
                    squared_error = (prediction - y[i]) ** 2
                    squared_error_list.append(squared_error)

                import statistics
                mse = statistics.mean(squared_error_list)
                # fitness
                mean_fitness_currchrom = 1000 / (1 + mse)

                return mean_fitness_currchrom

            elif fitness_func == 'r2':
                prediction_list = []
                for i in range(len(pd.DataFrame(x))):
                    variable_dict = {}
                    nth_input = 0
                    for term in self.term_set:
                        if term != '?':
                            variable_dict[term] = pd.DataFrame(x).iloc[i, nth_input]
                            nth_input += 1

                    prediction = EvaluateET(chromosome, variable_dict)
                    prediction_list.append(prediction)

                r2_result = r2_score(y, prediction_list) * 1000
                if r2_result <= 0:
                    r2_result = 1e-6

                return r2_result


        def InitializePopulation(popsize):
            '''Initialize a list of population sized popsize randomly'''

            population = []

            for i in range(popsize):
                # create head
                chromosome = []
                for i in range(self.nhead):
                    chromosome.append(random.choice(self.func_set + self.term_set))

                # create tail
                for i in range(self.ntail):
                    chromosome.append(random.choice(self.term_set))

                # concatenate head+tail
                #chromosome = list(head +
                for i in range(self.dc_length):
                    chromosome.append(str(random.randint(0,self.dc_length-1)))

                # add to population
                population.append(chromosome.copy())

            return population

        def Selection(population, fitness):
            '''Perform selection by roulette wheel'''
            new_population = random.choices(population, weights=fitness, k=len(population))  # weighted random choice
            return new_population

        def Replication(population):
            '''Perform replication(not necessary)'''
            new_population = population
            return new_population

        def Mutation(population, probability):
            '''Perform one point Mutation'''
            import random
            new_population = []

            for chromosome in population:
                mutate = random.random() < probability
                if mutate==True:

                    index = random.randint(0, self.chrom_length - 1)
                    if index < self.nhead:  # if randomizer picks to mutate head region
                        chromosome[index] = random.choice(self.func_set + self.func_set)
                    elif index >= self.nhead:  # if randomizer picks to mutate tail region
                        chromosome[index] = random.choice(self.term_set)
                    #Mutation for constant domain
                    index_dc = -random.randint(1, self.dc_length)
                    chromosome[index_dc] = str(random.randint(0,self.dc_length-1))
                new_population.append(chromosome.copy())

            return new_population

        def Inversion(population, probability):
            '''Perform inversion at head'''
            import random
            new_population = []

            for chromosome in population:
                inverse = random.random() < probability
                if inverse==True:
                    # Pick sequence to be inverted by picking random index
                    indexes = sorted(random.sample(range(0, self.nhead - 1), 2))
                    start_index = indexes[0]
                    end_index = indexes[1]

                    inverse_seq = chromosome[start_index:end_index + 1]

                    # Create inverted sequence
                    inverted_seq = []
                    for element in reversed(inverse_seq):
                        inverted_seq.append(element)

                    # Plug in and replace inverted sequence to the chromosome
                    new = 0  # nth element in the inverted sequence
                    for i in range(start_index, end_index + 1):
                        chromosome[i] = inverted_seq[new]
                        new += 1

                    #add inversion for constant domain !!
                    indexes_dc = sorted(random.sample(range(self.chrom_length, self.chrom_length+self.dc_length), 2))
                    start_index_dc = indexes_dc[0]
                    end_index_dc = indexes_dc[1]

                    inverse_seq_dc = chromosome[start_index_dc:end_index_dc + 1]
                    inverted_seq_dc = []
                    for element in reversed(inverse_seq_dc):
                        inverted_seq_dc.append(element)
                    new_dc = 0
                    for i in range(start_index_dc, end_index_dc + 1):
                        chromosome[i] = inverted_seq_dc[new_dc]
                        new_dc += 1


                new_population.append(chromosome.copy())

            return new_population

        def ISTransposition(population, probability):
            '''Perform IS Transposition short seq from tail to head'''
            new_population = []
            for chromosome in population:
                transpose = random.random() < probability
                if transpose == True:
                    max_seq_len = self.ntail // 3

                    if self.ntail==1:
                        seq_len = 1
                    else:
                        seq_len = random.randint(1, max_seq_len)

                    start_index = random.randint(self.nhead, self.chrom_length-seq_len)
                    end_index = start_index + seq_len-1

                    transpose_seq = chromosome[start_index:end_index + 1]

                    insert_start_index = random.randint(1, self.nhead-1)

                    new = 0
                    i = insert_start_index
                    for counter in range(len(transpose_seq)):
                        chromosome[i] = transpose_seq[new]
                        new += 1
                        i += 1

                new_population.append(chromosome.copy())

            return new_population

        def RISTransposition(population, probability):
            '''Perform Root IS Transposition'''
            new_population = []
            for chromosome in population:
                transpose = random.random() < probability
                if transpose == True:
                    max_seq_len = self.nhead // 3
                    seq_len = random.randint(1, max_seq_len)

                    isFunction = False
                    trial = 0
                    while isFunction==False:
                        start_index = random.randint(1, self.nhead-1)
                        if chromosome[start_index] in self.func_set:
                            break
                        if trial>10: #check if no function at all in head, break
                            break
                        trial += 1

                    end_index = start_index + seq_len-1

                    transpose_seq = chromosome[start_index:end_index + 1]

                    i = 0
                    for counter in range(len(transpose_seq)):
                        chromosome[i] = transpose_seq[i]
                        i+=1

                new_population.append(chromosome.copy())


            return new_population

        def GeneTransposition(population):
            '''Perform replication'''
            new_population = population
            return new_population

        def OnePointRecombination(population, probability):
            '''Perform OnePoint Recombination'''
            new_population = []
            recombination_pool = []
            for chromosome in population:  #Choose chromosomes to recombine
                recombine = random.random() < probability
                if recombine == True:
                    recombination_pool.append(chromosome.copy())
                elif recombine == False:
                    new_population.append(chromosome.copy())

            if len(recombination_pool) == 1: #If only 1 to recombine, return it
                new_population.append(recombination_pool[0].copy())

            while len(recombination_pool)>1:
                if len(recombination_pool)>2: #Determine which chromosome by index to recombine
                    indexes = sorted(random.sample(range(0, len(recombination_pool) - 1), 2))
                    first_parent = recombination_pool[indexes[0]].copy()
                    second_parent = recombination_pool[indexes[1]].copy()
                    recombination_pool.pop(indexes[0])
                    recombination_pool.pop(indexes[1] - 2)

                elif len(recombination_pool)==2:
                    first_parent = recombination_pool[0]
                    second_parent = recombination_pool[1]

                #Head and tail domain recombination
                recombination_start_index = random.randint(1, self.chrom_length - 2)
                #recombination_start_index = random.randint(1, len(chromosome)-2)
                first_child = first_parent[0:recombination_start_index]
                second_child = second_parent[0:recombination_start_index]

                first_cross = first_parent[recombination_start_index:]
                second_cross = second_parent[recombination_start_index:]

                for element in second_cross:
                    first_child.append(element)
                for element in first_cross:
                    second_child.append(element)

                #Constant domain recombination
                recombination_start_index_dc = random.randint(self.chrom_length+1, self.chrom_length+self.dc_length-2)
                first_child = first_parent[0:recombination_start_index_dc]
                second_child = second_parent[0:recombination_start_index_dc]

                first_cross = first_parent[recombination_start_index_dc:]
                second_cross = second_parent[recombination_start_index_dc:]

                for element in second_cross:
                    first_child.append(element)
                for element in first_cross:
                    second_child.append(element)
                ######

                new_population.append(first_child.copy())
                new_population.append(second_child.copy())

                if len(recombination_pool) == 1:
                    new_population.append(recombination_pool[0].copy())
                    break
                elif len(recombination_pool)==2:
                    break

            return new_population

        def TwoPointRecombination(population, probability):
            '''Perform replication'''
            new_population = []
            recombination_pool = []
            for chromosome in population:
                recombine = random.random() < probability
                if recombine == True:
                    recombination_pool.append(chromosome.copy())
                elif recombine == False:
                    new_population.append(chromosome.copy())

            if len(recombination_pool) == 1:
                new_population.append(recombination_pool[0].copy())

            while len(recombination_pool)>1:
                if len(recombination_pool)>2:
                    indexes = sorted(random.sample(range(0, len(recombination_pool) - 1), 2))
                    first_parent = recombination_pool[indexes[0]].copy()
                    second_parent = recombination_pool[indexes[1]].copy()
                    recombination_pool.pop(indexes[0])
                    recombination_pool.pop(indexes[1] - 2)

                elif len(recombination_pool)==2:
                    first_parent = recombination_pool[0]
                    second_parent = recombination_pool[1]

                recombination_indexes = sorted(random.sample(range(1, len(chromosome) - 2), 2))

                recombination_start_index = recombination_indexes[0]
                recombination_end_index = recombination_indexes[1]

                first_end_original = first_parent[recombination_end_index+1:]
                second_end_original = second_parent[recombination_end_index+1:]

                first_child = first_parent[0:recombination_start_index]
                second_child = second_parent[0:recombination_start_index]

                first_cross = first_parent[recombination_start_index:recombination_end_index+1]
                second_cross = second_parent[recombination_start_index:recombination_end_index+1]


                for element in second_cross:
                    first_child.append(element)
                for element in first_cross:
                    second_child.append(element)

                for element in first_end_original:
                    first_child.append(element)
                for element in second_end_original:
                    second_child.append(element)

                new_population.append(first_child.copy())
                new_population.append(second_child.copy())

                if len(recombination_pool) == 1:
                    new_population.append(recombination_pool[0].copy())
                    break
                elif len(recombination_pool)==2:
                    break

            return new_population

        def GeneRecombination(population):
            '''Perform replication'''
            new_population = population
            return new_population
            ########################################################################################

        print(f'''
=========================================================
Starting Gene Expression Programming Process
Population size:{popsize}
Generations:{ngenerations}
Function set: {self.func_set}
Terminal set: {self.term_set}
Chromosome length: {self.chrom_length}
Constant list: {self.const_list}
=========================================================
        ''')

        self.ngenerations = ngenerations
        # Every Chromosome is a string, population is a list of strings
        population = InitializePopulation(popsize).copy()  # Initialize a population
        print('Population initialized')
        # print(population)
        generation = 0  # Initialize at generation 0

        while generation <= ngenerations:

            # Perform EvalFitness on every chromosome on current generation's population
            fitness = [EvalFitness(chromosome, x, y, fitness_func) for chromosome in population].copy()
            fitness_mse = list((1000/np.array(fitness.copy()))-1).copy()
            print(f'Gen:{generation} Fitness Calculation completed')

            self.gen_pop_fit_history = AddGenInfo(self.gen_pop_fit_history, generation, population,
                                                  fitness, fitness_mse)  # Update history library

            gen_fittest = self.gen_pop_fit_history[generation]['Fittest Chromosome'].copy()
            gen_fittest = ''.join(gen_fittest)
            fittest_value = self.gen_pop_fit_history[generation]['Max Fitness Value']
            fittest_mse = self.gen_pop_fit_history[generation]['Fittest mse']
            print(f'Gen:{generation} Fittest chromosome:({gen_fittest})     Fitness value:{fittest_value}       MSE:{fittest_mse}')
            #print(self.gen_pop_fit_history)

            if generation==ngenerations:  # break while loop if last generation or (gen==ngen)
                break

            ## Entering new generation process
            # Selection (to choose parents and construct parent populations to reproduce)
            population = Selection(population, fitness)
            print(
                f'--------------------------------------------------------\nGen:{generation + 1}\nSelection from generation {generation} for generation {generation + 1} completed')
            # Replication
            population = Replication(population)

            print(f'Gen:{generation + 1} Reproduction process begin !')
            # Mutation
            population = Mutation(population, self.operator_probabilities['Mutation'])
            print(f'Gen:{generation + 1} Mutation completed')
            # Inversion
            population = Inversion(population, self.operator_probabilities['Inversion'])
            print(f'Gen:{generation + 1} Inversion completed')
            # IS Transposition
            population = ISTransposition(population, self.operator_probabilities['IS Transposition'])
            print(f'Gen:{generation + 1} IS Transposition completed')
            # RIS Transposition
            population = RISTransposition(population, self.operator_probabilities['RIS Transposition'])
            print(f'Gen:{generation + 1} RIS Transposition completed')
            # Gene Transposition
            population = GeneTransposition(population)
            # One-point Recombination
            population = OnePointRecombination(population, self.operator_probabilities['One-point Recombination'])
            print(f'Gen:{generation + 1} One-point Recombination completed')
            # Two-point Recombination
            population = TwoPointRecombination(population, self.operator_probabilities['Two-point Recombination'])
            print(f'Gen:{generation + 1} Two-point Recombination completed')
            # Gene Recombination
            population = GeneRecombination(population)

            print(f'Gen:{generation + 1} Reproduction process done !')

            # Perform elitism
            population[0] = self.gen_pop_fit_history[generation]['Fittest Chromosome']
            print(f'Elitism: Gen {generation} --> Gen {generation + 1}')

            generation += 1  # generation +1 End of a new generation
        final_fittest_list = self.gen_pop_fit_history[ngenerations]['Fittest Chromosome']
        final_fittest_ET = ChromToET(final_fittest_list)
        final_fittest = ''.join(final_fittest_list)
        final_fitness = self.gen_pop_fit_history[ngenerations]['Max Fitness Value']
        print(f'''
=========================================================
Completed Gene Expression Programming Process
Fittest Chromosome Result:({final_fittest}) with fitness value {final_fitness}
=========================================================
        ''')
        f = open("result.txt","w+")
        f.write(f"After {generation} generations, fittest Chromosome Result:({final_fittest}) with fitness value {final_fitness}\n in list {final_fittest_list}\n constant list: {list(self.const_list)}\nExpression Tree: {final_fittest_ET}")

