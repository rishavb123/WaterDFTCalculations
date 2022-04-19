import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
import ase.io
from ase.calculators.emt import EMT
from ase.build import molecule


from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer

# get training images by reading trajectory files

# read all images from the trajectory
training = ase.io.read("./training_data.traj", index=":")

# read every 10th image from the trajectory
# training = ase.io.read("./data/water_2d.traj", index="::10")

# print the length of the image
print(f"Length of training: {len(training)}")

# cell size
print(training[0].get_cell())

# periodic boundary condition
print(training[0].get_pbc())

# atomic positions
print(training[0].get_positions())

# system potential energy
print(training[0].get_potential_energy())

# forces
print(training[0].get_forces())

# ase.Atoms object
print(training[0])

# define sigmas
nsigmas = 10
sigmas = np.linspace(0, 2.0,nsigmas+1,endpoint=True)[1:]
print(sigmas)

# define MCSH orders
MCSHs_index = 2
MCSHs_dict = {
    0: { "orders": [0], "sigmas": sigmas,},
    1: { "orders": [0,1], "sigmas": sigmas,},
    2: { "orders": [0,1,2], "sigmas": sigmas,},
    3: { "orders": [0,1,2,3], "sigmas": sigmas,},
    4: { "orders": [0,1,2,3,4], "sigmas": sigmas,},
    5: { "orders": [0,1,2,3,4,5], "sigmas": sigmas,},
    6: { "orders": [0,1,2,3,4,5,6], "sigmas": sigmas,},
    7: { "orders": [0,1,2,3,4,5,6,7], "sigmas": sigmas,},
    8: { "orders": [0,1,2,3,4,5,6,7,8], "sigmas": sigmas,},
    9: { "orders": [0,1,2,3,4,5,6,7,8,9], "sigmas": sigmas,},
}
MCSHs = MCSHs_dict[MCSHs_index] # MCSHs is now just the order of MCSHs. 


GMP = {
    "MCSHs": MCSHs,
    "atom_gaussians": {
        "H": "./valence_gaussians/H_pseudodensity_2.g",
        "O": "./valence_gaussians/O_pseudodensity_4.g",
    },
    "cutoff": 12.0,
    "solid_harmonics": True,
}

elements = ["H", "O"]

config = {
    "model": {
        "name":"singlenn",
        "get_forces": True,
        "num_layers": 3,
        "num_nodes": 10,
        "batchnorm": True,
        "activation":torch.nn.Tanh,
    },
    "optim": {
        "force_coefficient": 0.01,
        "lr": 1e-3,
        "batch_size": 16,
        "epochs": 500,
        "loss": "mse",
        "metric": "mae",
    },
    "dataset": {
        "raw_data": training,
        "fp_scheme": "gmpordernorm",
        "fp_params": GMP,
        "elements": elements,
        "save_fps": True,
        "scaling": {"type": "normalize", "range": (0, 1)},
        "val_split": 0.1,
    },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "test",
        "verbose": True,
        # Weights and Biases used for logging - an account(free) is required
        "logger": False,
    },
}

torch.set_num_threads(1)
trainer = AtomsTrainer(config)
trainer.train()

predictions = trainer.predict(training)

# assess errors

true_energies = np.array([image.get_potential_energy() for image in training])
pred_energies = np.array(predictions["energy"])

print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2))
print("Energy MAE:", np.mean(np.abs(true_energies - pred_energies)))

training[0].set_calculator(AMPtorch(trainer))
training[0].get_potential_energy()

# set up images with one changing bond length
distances = np.linspace(0.4, 2.0, 100)
images = []
for dist in distances:
    image = molecule("H2O", vacuum=10.0)
    image.set_cell([10, 10, 10])
    image.set_pbc([1, 1, 1])

    # change bond length
    image.set_distance(0, 2, dist)
    image.set_angle(1, 0, 2, 104.210)
    images.append(image)

predictions = trainer.predict(images)

# get training point

training_angle100 = [_ for _ in training if np.isclose(_.get_angle(1, 0, 2), 104.210, atol=1e-3)]

distances_training = [_.get_distance(0, 2) for _ in training_angle100]
energies_training = [_.get_potential_energy() for _ in training_angle100]

plt.scatter(distances, predictions["energy"], label="prediction")
plt.scatter(distances_training, energies_training, label="training")
plt.xlabel("O-H bond length [A]")
plt.ylabel("potential energy [eV]")
plt.legend()

plt.show()