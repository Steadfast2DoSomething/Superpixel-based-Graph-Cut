from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
 
from sklearn.datasets import load_boston
boston = load_boston()
rf = RandomForestRegressor()
rf.fit(boston.data[:300], boston.target[:300])

print len(boston.data)
instances = boston.data[[300, 309]]

#print boston.data[300]
#print boston.data[309]
#print instances
#print instances[0].reshape(-1, 1).transpose()
#print instances[1].tolist()

#print "Instance 0 prediction:", rf.predict(instances[0].reshape(-1, 1).transpose())
#print "Instance 1 prediction:", rf.predict(instances[1].reshape(-1, 1).transpose())

prediction, bias, contributions = ti.predict(rf, instances)

print "prediction", prediction
for i in range(len(instances)):
    print "Instance", i + 1
    print "Bias (trainset mean)", bias[i]
    print "Feature contributions:"
    for c, feature in sorted(zip(contributions[i], boston.feature_names), key=lambda x: -abs(x[0])):
        print feature, round(c, 2)
    print "-" * 20 
