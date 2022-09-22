import pandas as pd
import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv('final_data.csv')
df.drop(['Unnamed: 0'],axis=1,inplace=True)

df['Radius']=df['Radius'].apply(lambda x: x.replace('$', '').replace(',', '')).astype('float')

radius = df['Radius'].to_list()
mass = df['Mass'].to_list()
gravity =[]

#converting solar mass and radius into km & kg
def convert_to_si(radius,mass):
    for i in range(0,len(radius)-1):
        radius[i] = radius[i]*6.957e+8
        mass[i] = mass[i]*1.989e+30
        
convert_to_si(radius,mass)


def gravity_calculation(radius,mass):
    G=6.674e-11
    for index in range(0,len(mass)):
        g=(mass[index]*G)/((radius[index])**2)
        gravity.append(g)
gravity_calculation(radius,mass)
df["Gravity"]=gravity
df

mass = df["Mass"].to_list()
radius = df["Radius"].to_list()
dist = df["Distance"].to_list()
gravity = df["Gravity"].to_list()

df.to_csv("star_with_gravity.csv")

df = pd.read_csv("star_with_gravity.csv")

mass.sort()
radius.sort()
gravity.sort()
plt.plot(radius,mass)


plt.title("Radius & Mass of the Star")
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

plt.plot(mass,gravity)

plt.title("Mass vs Gravity")
plt.xlabel("Mass")
plt.ylabel("Gravity")
plt.show()


plt.scatter(radius,mass)
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

df.to_csv("star_with_gravity.csv")


X = df.iloc[:,[3,4]].values

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append((kmeans.inertia_))
plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel('Number of clusters')
plt.show()

bools=[]
for d in df.Distance:
    if d<=100:
        bools.append(True)
    else:
        bools.append(False)

is_dist=pd.Series(bools)

is_dist.head()

stars_dist=df[is_dist]
stars_dist.reset_index(inplace=True,drop=True)
stars_dist.head()

gravity_bool=[]
for g in stars_dist.Gravity:
    if g<=350 and g>=150:
        gravity_bool.append(True)
    else:
        gravity_bool.append(False)

is_gravity = pd.Series(gravity_bool)
final_stars = stars_dist[is_gravity]
final_stars.head()
final_stars.reset_index(inplace=True,drop=True)
final_stars.head()

final_stars.to_csv("filtered_stars.csv")