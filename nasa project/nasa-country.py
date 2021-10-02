from sklearn.ensemble import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import *
from pandas import DataFrame
df = pd.read_csv('nasaa.csv')

aaa = np.array(DataFrame.drop_duplicates(df[['Country']]))
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

def number_country_change(x):
    if x == 'United States':
        return 0
    elif x == 'Canada':
        return 1
    elif x == 'Palau':
        return 2
    elif x == 'Ecuador':
        return 3
    elif x == 'Micronesia, Federated States of':
        return 4
    elif x == "Bahamas":
        return 5
    elif x == "Mexico":
        return 6
    else:
        return 7

df['Country_Change'] = df['Country'].apply(number_country_change)

y = df.loc[:,['Country_Change']]

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

#y['Country_Change'] = change.fit_transform(y['Country'])
#y['Date_Change'] = change.fit_transform(y['Date'])
#y['State_Change'] = change.fit_transform(y['State'])
#y['County_Change'] = change.fit_transform(y['County'])
#y['Country_Change'] = change.fit_transform(y['Country'])

y_n = y[['Country_Change']]

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
    X_train, X_test, y_train, y_test = train_test_split(X.values, y_n.values, test_size=0.2,
                                                        stratify=None,
                                                        shuffle=True,
                                                        random_state=36)

    model_nasa_emirhan = ExtraTreesClassifier(criterion="gini",
                                              max_depth=None,
                                              max_features="auto",
                                              random_state=84,
                                              n_estimators=10,
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

# feature names are loading..
feature_name = X.columns
print(y.columns)
# import is the export graphviz function at tree in sklearn :))
from sklearn.tree import export_graphviz


# tree classification creating of the visulation or graph!
def save_decision_trees_as_dot(model_nasa_emirhan,feature_name, iteration):
    file_name = open("country_classification_nasa_emirhan" + str(iteration) + ".dot", 'w')
    dot_data = export_graphviz(
        model_nasa_emirhan,
        out_file=file_name,
        feature_names=feature_name,
        class_names=['United States', ' Canada', ' Palau', ' Ecuador', ' Micronesia', ' Federated States of', ' Bahamas', ' Mexico', ' Tonga'],
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

#print(DataFrame.drop_duplicates(y))

#Country Label = Accuracy Score: 0.9788235294117648
#Country Label = Precision Score: 0.978506841585555
#Country Label = Recall Score: 0.9788235294117648
#Country Label = F1 Score: 0.9783583602026715