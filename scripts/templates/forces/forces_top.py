import os 
import sys
from pathlib import Path
import subprocess
import shutil   
import h5py


#Define
root_directory = "/data/horse/ws/miha493e-research_project/activediffusion"
exp = "full_dataset"
prop= "eMBD"
num_targets = 4
num_models = 4

#Run
evaluation_data = f"{root_directory}/data/data_{exp}/evaluation"
for m in range(num_models):
    for t in range(num_targets):
        extra_properties_file = f"{evaluation_data}/target{t}/model{m}/hdf5/m{m}-t{t}-extra_properties.hdf5"
        output_file = f"{evaluation_data}/target{t}/model{m}/hdf5/m{m}-t{t}-output.hdf5"
        accuracy_file = f"{evaluation_data}/target{t}/model{m}/results/{prop}.csv"
        data={}
        with open(accuracy_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                mol, _, _, _, relErr = parts
                data[mol] = abs(float(relErr))
        # select lowest 20 relative error molecules and remove .xzy from mol names
        sorted_data = sorted(data.items(), key=lambda x: x[1])
        top_20_mols = sorted_data[:20]
        top_20_mol_names = [mol[:-4] for mol, _ in top_20_mols]

        # copy data for top 20 molecules to a new directory
        top_20_extra_prop = f"{evaluation_data}/target{t}/model{m}/hdf5/top_20-{prop}-m{m}-t{t}-extra_properties.hdf5"
        top_20_output = f"{evaluation_data}/target{t}/model{m}/hdf5/top_20-{prop}-m{m}-t{t}-output.hdf5"
        
        # remove existing files if they exist
        if os.path.exists(top_20_extra_prop):
            os.remove(top_20_extra_prop)
        if os.path.exists(top_20_output):
            os.remove(top_20_output)

        with h5py.File(extra_properties_file, "r") as f:
            with h5py.File(top_20_extra_prop, "w") as f_out:
                for group_name in f.keys():
                    grp = f_out.require_group(f"Top20-{prop}-{group_name}")
                    for mol in f[group_name].keys():
                        if mol not in top_20_mol_names:
                            continue
                        mol_group = grp.create_group(mol)
                        mol_group.attrs.update(f[f"{group_name}/{mol}"].attrs)
                        for key in f[group_name][mol].keys():
                            f.copy(f"{group_name}/{mol}/{key}", mol_group, name=key)

        with h5py.File(output_file, "r") as f:
            with h5py.File(top_20_output, "w") as f_out:
                for group_name in f.keys():
                    grp = f_out.require_group(f"Top20-{prop}-{group_name}")
                    for mol in f[group_name].keys():
                        if mol not in top_20_mol_names:
                            continue
                        f.copy(f"{group_name}/{mol}", grp, name=mol)


