import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4])
y = np.array([4,3,2,1])


plt.figure()
plt.plot(x,y,color="r",alpha=0.7,label="line")
plt.scatter(x,y,color="b",alpha=0.4,label="scatter")
plt.title("Matplotlib")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.xticks([0,1,2,3,4,5])
plt.legend()
plt.show()


fig, axes = plt.subplots(2,1, figsize=(9,7))
fig.subplots_adjust(hspace = 0.5)

x = list(range(1,11))
y = list(range(10,0,-1))
print(y)

axes[0].scatter(x,y)
axes[0].set_title("sub-1")
axes[0].set_ylabel("sub-1 y")
axes[0].set_xlabel("sub-1 x")

axes[1].scatter(y,x)
axes[1].set_title("sub-2")
axes[1].set_ylabel("sub-2 y")
axes[1].set_xlabel("sub-2 x")
plt.show()

plt.figure()
img = np.random.random((50,50))
plt.imshow(img, cmap = "gray") # 0 (siyah) 1 (beyaz) --> 0.5 (gri)
plt.axis("off")
plt.show()

# %%{ os
import os

print(os.name)

currentDir = os.getcwd()
print(currentDir)

# new folder
folder_name = "new_folder"
os.mkdir(folder_name)

new_folder_name = "new_folder_2"
os.rename(folder_name,new_folder_name)

os.chdir(currentDir+"\\"+new_folder_name)
print(os.getcwd())

os.chdir(currentDir)
print(os.getcwd())

files = os.listdir()
print(files)

for f in files:
    if f.endswith(".py"):
        print(f)


os.rmdir(new_folder_name)

for i in os.walk(currentDir):
    print(i)

os.path.exists("matplotlipKütüphanesi.py")

