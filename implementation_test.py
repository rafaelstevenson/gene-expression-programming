import pandas as pd
from GEP import GeneExpressionProgramming


df = pd.read_excel('../testing_datasets/sqrt_test_func_dataset_plusonepointone.xls')

#func_set = ['+','-','*','/', '(sqrt)']
func_set = ['+','*','(sqrt)','-']
term_set = ['a','b','?']
const_range = [0.1,3] #inclusive ends
operator_probabilities = {
    "Mutation":0.3, "Inversion":0.1, "IS Transposition":0.1,
    "RIS Transposition":0.1, "One-point Recombination":0.3,
    "Two-point Recombination":0.3
}

head_length = 7
population_size = 300
generations = 200
fitness_func = 'mse'

GEPProcess = GeneExpressionProgramming(head_length,func_set,term_set,const_range, operator_probabilities)
GEPProcess.RunGEP(df[['input1','input2']],df['output'],population_size,generations,fitness_func)
GEPProcess.VisualizeResults()