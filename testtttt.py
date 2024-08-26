import torch
import matplotlib.pyplot as plt
import numpy as np

def mutate_parameter(param, mutation_prob=1.0, mutation_strength=0.1):
    if torch.rand(1).item() < mutation_prob:
        mutation = torch.randn_like(param) * mutation_strength
        param.data.add_(mutation)

def simulate_mutations(initial_value, mutation_strengths, iterations=100):
    results = {}
    initial_tensor = torch.tensor([initial_value], dtype=torch.float32)
    
    for strength in mutation_strengths:
        values = [initial_tensor.item()]
        param = initial_tensor.clone()
        
        for _ in range(iterations):
            mutate_parameter(param, mutation_prob=1.0, mutation_strength=strength)
            values.append(param.item())
        
        results[strength] = values
    
    return results

def plot_simulation(results):
    plt.figure(figsize=(12, 6))
    
    for strength, values in results.items():
        plt.plot(values, label=f'Mutation Strength: {strength}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Value Over Time with Different Mutation Strengths')
    plt.legend()
    plt.grid(True)
    plt.show()

# Parameters for simulation
initial_value = 0.0
mutation_strengths = [0.01, 0.05, 0.1, 0.2]
iterations = 10000

# Run the simulation
results = simulate_mutations(initial_value, mutation_strengths, iterations)

# Plot the results
plot_simulation(results)
