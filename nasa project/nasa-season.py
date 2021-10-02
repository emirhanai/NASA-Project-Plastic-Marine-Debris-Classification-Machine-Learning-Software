from sklearn.ensemble import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import *
from pandas import DataFrame
df = pd.read_csv('nasaa.csv')

aaa = np.array(DataFrame.drop_duplicates(df[['Season']]))
bbb = np.array2string(aaa)
ccc = bbb.replace("[","")
ddd = ccc.replace("]","")
eee = ddd.replace("\n",",")
fff = eee.replace("'","")
ggg = fff.replace('"',"")
#print(ggg.split(","))
X = df.iloc[:,33:140]

#y = df.loc[:,['Survey_Type','Date','Country']]

#y = df.loc[:,['Country']]

y = df.loc[:,['Season']]

#print(y)

from pandas import DataFrame

a = np.array(DataFrame.drop_duplicates(y))
b = np.array2string(a)
c = b.replace("[", "")
d = c.replace("]", "")
e = d.replace("\n", ",")
g = e.replace('"', "")
f = g.replace("'","")
h = f.split(",")
#print(ff)


#print(y.duplicated())
change = LabelEncoder()

y['Season_Change'] = change.fit_transform(y['Season'])
#y['Date_Change'] = change.fit_transform(y['Date'])
#y['State_Change'] = change.fit_transform(y['State'])
#y['County_Change'] = change.fit_transform(y['County'])
#y['Country_Change'] = change.fit_transform(y['Country'])

y_n = y.drop(['Season'],axis='columns')

aa = np.array(DataFrame.drop_duplicates(y))
bb = np.array2string(aa)
cc = bb.replace("[", "")
dd = cc.replace("]", "")
ee = dd.replace("\n", ",")
gg = ee.replace('"', "")
ff = gg.replace("'","")
hh = ff.split(",")
#print(hh)
#print(h)

#print(y_n)

#print(X)
#print(X_n.shape)

#print(y)

for i in np.arange(1,2,1):
    X_train, X_test, y_train, y_test = train_test_split(X.values, y_n.values, test_size=0.004,
                                                        stratify=None,
                                                        shuffle=True,
                                                        random_state=16)

    model_nasa_emirhan = ExtraTreesClassifier(criterion="gini",
                                              max_depth=None,
                                              max_features="auto",
                                              random_state=37,
                                              n_estimators=26,
                                              n_jobs=-1,
                                              verbose=0,
                                              class_weight="balanced")

    from sklearn.multioutput import MultiOutputClassifier

    model_nasa_emirhan.fit(X_train, y_train)

    pred_nasa = model_nasa_emirhan.predict(X_test)

    from sklearn.metrics import *

    print(accuracy_score(y_test, pred_nasa),"x",i)
    print(precision_score(y_test, pred_nasa, average='weighted'))
    print(recall_score(y_test, pred_nasa, average='weighted'))
    print(f1_score(y_test, pred_nasa, average='weighted'))

print(DataFrame.drop_duplicates(y))

#Survey Type Label = Accuracy Score: 1.0
#Survey Type Label = Precision Score: 1.0
#Survey Type Label = Recall Score: 1.0
#Survey Type Label = F1 Score: 1.0

# feature names are loading..

feature_name = X.columns

#print(y.columns)

# import is the export graphviz function at tree in sklearn :))
from sklearn.tree import export_graphviz


# tree classification creating of the visulation or graph!
def save_decision_trees_as_dot(model_nasa_emirhan,feature_name, iteration):
    file_name = open("season_classification_nasa_emirhan" + str(iteration) + ".dot", 'w')
    dot_data = export_graphviz(
        model_nasa_emirhan,
        out_file=file_name,
        feature_names=feature_name,
        class_names=['Fall','Spring','Summer','Winter'],
        rounded=True,
        proportion=False,
        precision=2,
        filled=True, )
    file_name.close()
    print("Extra Trees in forest :) {} saved as dot file".format(iteration + 1))

 # Crate of tree graph about of text classification :))
for i in range(len(model_nasa_emirhan.estimators_)):
    save_decision_trees_as_dot(model_nasa_emirhan.estimators_[1],feature_name,i)
    print(i)

#Season Label = Accuracy Score: 0.8974358974358975
#Season Label = Precision Score: 0.9098865877712031
#Season Label = Recall Score: 0.8974358974358975
#Season Label = F1 Score: 0.8973039731660422