import pandas as pd
import os

id = pd.read_csv("./result/id.csv")
label = pd.read_csv("./result/label.csv")

label = label[0:1040]

labels={
    0:'bedroom', 1:'coast', 2:'forest', 3:'highway', 4:'insidecity', 5:'kitchen', 6:'livingroom', 7:'mountain', 8:'office', 9:'opencountry',
     10:'street', 11:'suburb', 12:'tallbuilding'
}

for i in range(0, len(label)):
    index = int(label.iloc[i])
    label.iloc[i] = labels[index]
	
prediction = pd.concat([id, label], axis=1)
prediction.to_csv("./result/prediction2.csv", encoding="utf_8_sig", index=False)
