e=1.602E-19
m=9.109E-31
h=6.626E-34
hbar=1.055E-34
eps=8.854E-12
c=2.998E8
import numpy as np
E=16*h*c*(4*np.pi*eps*hbar)**2/(3*e**4*m)
print(E)

r=4*hbar**2*np.pi*eps/ (56*m*e**2)
print(r)

Em = 3/4*m*207/208*e**4 / (2*(4*np.pi*eps*hbar)**2)

l = h*c/Em

print(l)