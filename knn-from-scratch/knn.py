import math 
import operator
from collections import Counter

def euclidean_distance(a,b):
    distance = 0
    for i in range(len(a)):
        distance += (a[i] - b[i]) ** 2
        
    return math.sqrt(distance)

def get_neighbors(X,y,sample,k):
    distances = []
    for i in range(k):
        dist = euclidean_distance(sample,X.iloc[i].values)
        label = y.iloc[i]
        distances.append((X.iloc[i].values,label,dist))
        
    distances.sort(key=operator.itemgetter(2))
    return distances[:k]

def predict(X,y,sample,k):
    neighbors = get_neighbors(X,y,sample,k)
    
    labels = [n[1] for n in neighbors]
    vote = Counter(labels).most_common(1)[0][0]
    
    return vote