# Create continuous plots with lines of best fit
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = {
    'N': [2, 3, 4],
    'trial': [1, 1, 1],
    'iteration_reached': [0, 1027, 6419],
    'runtime_seconds': [6.23, 22664.68, 190343.17],
    'top_score': [6.0, 15.0, 28.0]
}

df = pd.DataFrame(data)
df['iterations_per_second'] = df['iteration_reached'] / df['runtime_seconds']

# Convergence plot with trend line
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='N', y='iteration_reached', scatter_kws={'s': 50}, line_kws={'color': 'red'})
plt.title('Convergence Plot with Trend Line')
plt.show()

# Runtime plot with trend line
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='N', y='runtime_seconds', scatter_kws={'s': 50}, line_kws={'color': 'red'})
plt.title('Runtime Scaling with Trend Line')
plt.show()

# Efficiency plot with trend line
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='N', y='iterations_per_second', scatter_kws={'s': 50}, line_kws={'color': 'red'})
plt.title('Efficiency Trend')
plt.show()

# Create plots
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='N', y='iteration_reached')
plt.title('Convergence Plot: Iterations vs N')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='N', y='runtime_seconds')
plt.title('Runtime vs N')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='N', y='iterations_per_second')
plt.title('Efficiency: Iterations per Second vs N')
plt.show()

# Calculate and display metrics
print("\
Summary Statistics:")
print(df.describe())

avg_time_per_iteration = df['runtime_seconds'].sum() / df['iteration_reached'].sum()
most_efficient_n = df.loc[df['iterations_per_second'].idxmax(), 'N']
fastest_convergence_n = df.loc[df['iteration_reached'].idxmin(), 'N']

print("\
Key Metrics:")
print(f"Average time per iteration: {avg_time_per_iteration:.2f} seconds")
print(f"Most efficient N: {most_efficient_n}")
print(f"Fastest convergence N: {fastest_convergence_n}")