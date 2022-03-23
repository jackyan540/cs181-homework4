import numpy as np
import matplotlib.pyplot as plt

x = [-3,-2,-1,0,1,2,3]
y = np.array([1,1,-1,1,-1,1,1])
basis = []
for i in x:
    basis.append((i,-(8/3)*(i**2)+(2/3)*(i**4)))
    
plt.show()
for i in range(len(y)):
    if y[i] == 1:
        plt.scatter(basis[i][0],basis[i][1],color='r')
    else:
        plt.scatter(basis[i][0],basis[i][1],color='b')
plt.scatter(-3,30,color='r',label='Positive Examples')
plt.scatter(-1,-2,color='b',label='Negative Examples')
plt.axhline(y=-1,label='Optimal Decision Boundary')
plt.title('Basis Transformed Training Data')
plt.xlabel('x')
plt.ylabel('-(8/3)x^2 + (2/3)x^4')
plt.legend()
plt.savefig('1_1.png')