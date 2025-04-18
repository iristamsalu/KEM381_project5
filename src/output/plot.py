import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('energy.dat',skiprows=1, delim_whitespace=True, header=None)
data.columns = ['index','Time', 'Potential Energy', 'Kinetic Energy', 'Total Energy', 'Temperature']

plt.figure(figsize=(15, 12))
plt.grid(True)
plt.plot(data['Time'], data['Potential Energy'], label='Potential Energy')
plt.plot(data['Time'], data['Kinetic Energy'], label='Kinetic Energy')
plt.plot(data['Time'], data['Total Energy'], label='Total Energy')
plt.show()
plt.xlabel("Time (fs)")
plt.ylabel("Energy (eV)")
plt.title("Energy vs Time")
plt.legend()
plt.savefig('energy_plot.png')
plt.show()
print(data)

