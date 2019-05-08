import pandas as pd

fold='test'


df = pd.read_csv("nih_labels.csv")
print(df.head(3))
df = df[df['fold'] == fold]
print(df.head(3))

starter_images = pd.read_csv("starter_images.csv")

print(df.size, starter_images.size)

df1= pd.merge(left=df, right=starter_images, how="inner", on="Image Index")
print(df1.head(3))
print(df1.size)
