import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

#ignoruojam betkokius ispejimus
warnings.filterwarnings('ignore')

#paleisti jupyter (terminale)
#python -m notebook

# paruosiam duomenu baze darbui
np.set_printoptions(suppress=True)
pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', 15)

#https://www.kaggle.com/kemical/kickstarter-projects
filename = 'ks-projects-201801.csv'
df = pd.read_csv(filename, encoding="iso8859_2") #ansi neveikia

# paziurim kaip atrodo
print(df)
df.info

#tikrinam ar yra null reiksmiu
df.isnull().any()
df.isnull().sum()

#pasalinam null reiskmes
df = df[df['name'].notnull() & df['usd pledged'].notnull()]
print(df)
df.isnull().sum()

#tikrinam kickstarteriu busenas, pasalinam tas, kurios nerupi
df['state'].value_counts()
df = df[df['state'].isin(['failed', 'successful'])]
df['state'].replace({'failed': 'Nesekmingi projektai', 'successful': 'Sekmingi projektai'}, inplace=True)
df['state'].value_counts()

#tikrinam ir sutvarkom datas, paversdami i datos duomenu tipa
df[['launched', 'deadline']].value_counts()
df[['launched', 'deadline']].dtypes
df = df.astype({'launched': 'datetime64'})
df = df.astype({'deadline': 'datetime64'})
df[['launched', 'deadline']].dtypes

#sukuriam stulpeli rodanti kiek liko iki deadline
df['laikas_iki_ks_pabaigos'] = df['deadline'] - df['launched']
df['laikas_iki_ks_pabaigos']

#padarom, kad rodytu tik likusias dienas (be laiko)
#trinam nereikalinga stulpeli
tik_dienos = df['laikas_iki_ks_pabaigos'].dt.days
df['dienos_iki_ks_pabaigos'] = tik_dienos
df['dienos_iki_ks_pabaigos']
df = df.drop('laikas_iki_ks_pabaigos', axis=1)

#country padarom US ir other
df['country'] = df['country'].replace(['GB', 'CA', 'AU', 'NO', 'IT', 'DE',
                                       'IE', 'MX', 'ES', 'SE', 'FR', 'NZ',
                                       'CH', 'AT', 'BE', 'DK', 'HK', 'NL',
                                       'LU', 'SG', 'JP'], 'other')
print(df)

#DARBAS SU DUOMENIM, DIAGRAMOS IR T.T.

#kiek sekmingu, kiek nesekmingu KS projektu
plt.figure(figsize=(7.5, 5))
plt.title('Kickstarter projektu sekmingumas', fontsize=16)
plt.pie(x=df['state'].value_counts(),
        labels=df['state'].value_counts().index,
        autopct='%1.1f%%', pctdistance=0.7,
        explode=(0, 0.1), startangle=73,
        shadow=True, colors={'red', 'green'})

#projektai pagal kategorijas ir ju kiekis
plt.figure(figsize=(15, 5))
plt.title('Kickstarter projektu kategorijos ir kiekis', fontsize=16)
plt.ylim(0, 60000)
sns.barplot(x=df['main_category'].value_counts().index,
            y=df['main_category'].value_counts(),
            palette='flare').set(ylabel=None)

#projektai pagal salis
plt.figure(figsize=(5, 3))
plt.title('Kickstarter projektu kiekis pagal salis', fontsize=16)
plt.ylim(0, 270000)
sns.barplot(x=df['country'].value_counts().index,
            y=df['country'].value_counts(),
            palette='viridis').set(ylabel=None)

#projektai pagal pinigu suma (brangiausi/pigiausi)
plt.figure(figsize=(17, 5))
plt.title('Projektu kategorijos, kurioms sukaupta daugiausia pinigu', fontsize=16)
plt.ylim(0, 22000000)
plt.scatter(df['main_category'], df['usd_pledged_real'], s=5, c='blue')
plt.grid()

#projektu laiko pasiskirstymas nuo pradzios iki deadline'o
ax1 = sns.displot(x='dienos_iki_ks_pabaigos', data=df, kind='kde', color='green')
ax1.set(xlabel='Dienos iki projekto pabaigos', ylabel='Kartojimosi daznis',
       title='Dienu iki projekto pabaigos pasikartojimas')

ax2 = sns.boxplot(x='dienos_iki_ks_pabaigos', data=df, color='green', linewidth=2, fliersize=2)
ax2.set(xlabel='Dienos iki projekto pabaigos',
       title='Dienu iki projekto pabaigos pasikartojimas')

#zmones, aukojantys projektams
ax3 = sns.boxplot(x='backers', data=df, color='blue', linewidth=1, fliersize=4)
ax3.set(xlabel='Projektams aukojanciu zmoniu kiekis',
       title='Projektams aukojanciu zmoniu pasiskirstymas')

#sumazinam aukojanciu zmoniu kieki ir sukauptu pinigu kieki, procentais
#zmones
df = df[df['backers'] < np.percentile(df['backers'], 97)]

#pinigai, nukerpam is apacios (maziausias sumas) ir is virsaus (didziausias)
df = df[(df['usd_pledged_real'] > np.percentile(df['usd_pledged_real'], 2)) &
        (df['usd_pledged_real'] < np.percentile(df['usd_pledged_real'], 98))]

#vidutinis pledge
df['Average_pledge'] = df.groupby(df['main_category'])['usd_pledged_real'].transform('mean')

#dar kart zvilgsnis i grafikus, su pakeistais duomenim:

#perdaryt i bar
#projektai pagal pinigu suma (brangiausi/pigiausi)
#vietoj sito plt bar
plt.figure(figsize=(17, 5))
plt.title('Projektu kategorijos, kurioms sukaupta daugiausia pinigu', fontsize=16)
plt.ylim(0, 40000)
plt.scatter(df['main_category'], df['usd_pledged_real'], s=5, c='blue')
plt.grid()

#bar vietoj scatter:
plt.figure(figsize=(15, 5))
plt.title('Projektu kategorijos pagal sukauptu pinigu vidurki', fontsize=16)
plt.ylim(0, 5500)
sns.barplot(x=df['main_category'].value_counts().index,
            y=df['Average_pledge'].unique(),
            palette='mako').set(ylabel=None)

#zmones, aukojantys projektams
ax3 = sns.boxplot(x='backers', data=df, color='blue', linewidth=1, fliersize=4)
ax3.set(xlabel='Projektams aukojanciu zmoniu kiekis',
       title='Projektams aukojanciu zmoniu pasiskirstymas')

#vel keiciu 'state' stulpeli. nes pagal ji skaiciuosiu ir kursiu modeli
df['state'].replace({'Nesekmingi projektai': '0', 'Sekmingi projektai': '1'}, inplace=True)
df['state'].value_counts()

#bandau sukurt logistine regresija:
#pasalinam nereikalinga stulpeli
df = df.drop('ID', axis=1)

xTrain, xTest, yTrain, yTest = train_test_split(df[['dienos_iki_ks_pabaigos',
                                                'usd_goal_real', 'backers']],
                                                df['state'], test_size=0.3, random_state=42)
xTrain = sm.add_constant(xTrain)
xTest = sm.add_constant(xTest)

sm_model = sm.Logit(yTrain.astype(float), xTrain.astype(float)).fit()
sm_model.summary()
df.corr()
sns.heatmap(df.corr())

y_estimates = round(sm_model.predict(xTest))
confusion_matrix(yTest.astype(float), y_estimates.astype(float), normalize='pred')
accuracy_score(yTest.astype(float), y_estimates.astype(float))
recall_score(yTest.astype(float), y_estimates.astype(float), average=None)
precision_score(yTest.astype(float), y_estimates.astype(float), average=None)

#Regresija kitaip:
X = df[['dienos_iki_ks_pabaigos', 'usd_goal_real', 'backers']]
Y = df['state']

train = df[:(int((len(df)*0.3)))]
test = df[(int((len(df)*0.3))):]

Log1 = LogisticRegression()

train_x = np.array(train[['dienos_iki_ks_pabaigos', 'usd_goal_real', 'backers']])
train_y = np.array(train['state'])

Log1.fit(train_x, train_y)

test_x = np.array(test[['dienos_iki_ks_pabaigos', 'usd_goal_real', 'backers']])
test_y = np.array(test['state'])

#spejimas
state_pred = Log1.predict(test_x)
R = r2_score(test_y, state_pred)
print('R: ', R)

#funkcija:
def speti():
    X = df[['dienos_iki_ks_pabaigos', 'usd_goal_real', 'backers']]
    Y = df['state']
    Log1 = LogisticRegression()
    Log1.fit(X, Y)
    t = input('Iveskite dienas iki projekto pabaigos: ')
    t2 = int(t)
    u = input('Iveskite reikalinga sukaupti pinigu suma: ')
    u2 = int(u)
    g = input('Iveskite reikalinga backers kieki: ')
    g2 = int(g)
    new = [[t2, u2, g2]]
    out = Log1.predict(new)
    return print('Dienos: ' + str(t2) + ', pinigu suma: ' + str(u2) + ', backers: ' +
                 str(g2) + '. Projekto sekme: ' + str(out) + ' (1 - sekmingas, '
                                                             '0 - ne).')
#kaimynai
xTrain, xTest, yTrain, yTest = train_test_split(df[['dienos_iki_ks_pabaigos',
                                                'usd_goal_real', 'backers']],
                                                df['state'], test_size=0.3, random_state=100)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(xTrain, yTrain)
yPred = knn.predict(xTest)
conf_m = confusion_matrix(yTest, yPred)
tikslumas = (conf_m[1, 0]+conf_m[0, 1])/(conf_m.sum())
print(tikslumas)

#ieskom geriausiu kaimynu:
x = list()
y = list()
for i in range(1, 20):
    xTrain, xTest, yTrain, yTest = train_test_split(df[['dienos_iki_ks_pabaigos',
                                                        'usd_goal_real', 'backers']],
                                                    df['state'], test_size=0.3, random_state=100)
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(xTrain, yTrain)
    yPred = knn.predict(xTest)
    conf_m = confusion_matrix(yTest, yPred)
    tikslumas = (conf_m[1, 0] + conf_m[0, 1]) / (conf_m.sum())
    x.append(tikslumas)
    y.append(i)
    print(x, y)

#desicion tree
xTrain, xTest, yTrain, yTest = train_test_split(df[['dienos_iki_ks_pabaigos',
                                                'usd_goal_real', 'backers']],
                                                df['state'], test_size=0.3, random_state=100)
model = DecisionTreeClassifier(max_depth=3)
model.fit(xTrain, yTrain)

yPred = model.predict(xTest)
conf_m = confusion_matrix(yTest, yPred)
tikslumas = (conf_m[1, 0]+conf_m[0, 1])/(conf_m.sum())
print(tikslumas)

#nupiesti:
plot_tree(model, fontsize=7)

#forest
xTrain, xTest, yTrain, yTest = train_test_split(df[['dienos_iki_ks_pabaigos',
                                                'usd_goal_real', 'backers']],
                                                df['state'], test_size=0.3, random_state=100)
model = RandomForestClassifier(max_depth=3)
model.fit(xTrain, yTrain)

yPred = model.predict(xTest)
conf_m = confusion_matrix(yTest, yPred)
tikslumas = (conf_m[1, 0]+conf_m[0, 1])/(conf_m.sum())
print(tikslumas)
