import pickle
import matplotlib.pyplot as plt

with open("Models/SMK_20211025", "rb") as file:
    d = pickle.load(file)

plt.plot(d)
plt.savefig("loss_smk.png")