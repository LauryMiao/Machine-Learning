#
# Notice: When choosing X[:[0,1]] or X[:[0,2]] as features, the following errors will hit. 
#         (Guess something not robust within mlxtend)
#
#   File "C:\Laury\ML\ml_exercises\kmeans.py", line 19, in <module>
#    km.fit(X)
#  File "C:\ProgramData\Anaconda3\lib\site-packages\mlxtend\_base\_cluster.py", line 39, in fit
#    self._fit(X=X, init_params=init_params)
#  File "C:\ProgramData\Anaconda3\lib\site-packages\mlxtend\cluster\kmeans.py", line 107, in _fit
#    n_iter=self.max_iter)
#  File "C:\ProgramData\Anaconda3\lib\site-packages\mlxtend\_base\_iterative_model.py", line 34, in _print_progress
#    ela_sec = time() - self.init_time_
#AttributeError: 'Kmeans' object has no attribute 'init_time_'
#

import matplotlib.pyplot as plt
from mlxtend.data import iris_data
from mlxtend.cluster import Kmeans

X, y = iris_data()
X = X[:,[2,3]]  # choose combination 2 of 4 features, then will results in 6 kinds of results. see the png.

#plt.scatter(X[:,0],X[:,1], c='red')
#plt.show()

km = Kmeans(k=3,
            max_iter=50,
            random_seed=1,
            print_progress=3)
km.fit(X)

print(':\nIterations until convergence:',km.iterations_)
print('Final centroids:\n',km.centroids_)


# Visualize the cluster memberships
y_cluster = km.predict(X)

plt.scatter(X[y_cluster == 0,0],
            X[y_cluster == 0,1],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1')

plt.scatter(X[y_cluster == 1,0],
            X[y_cluster == 1,1],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2')

plt.scatter(X[y_cluster == 2,0],
            X[y_cluster == 2,1],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3')

plt.scatter(km.centroids_[:,0],
            km.centroids_[:,1],
            s=250,
            c='red',
            marker='*',
            label='centroids')

plt.legend(loc='best',
            scatterpoints=1)
plt.grid()
plt.show()

