
# Read and analyze results.csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('results.csv')

# Print column names and first few rows to verify structure
print("Columns:", df.columns.tolist())
print("\
First few rows:")
print(df.head())
# Calculate efficiency metric and create continuous plots
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

# Print summary statistics
print("\
Summary Statistics:")
print(df[['N', 'iteration_reached', 'runtime_seconds', 'iterations_per_second']].describe())