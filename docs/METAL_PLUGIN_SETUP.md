# Installing GPU-Accelerated OpenMM (Metal) on Apple Silicon Macs

To achieve a dramatic speed-up in the Molecular Dynamics (MD) validation step, we will compile and install a plugin that allows the OpenMM simulation engine to use your Mac’s Metal GPU.

> **Note:** This is an advanced, optional step. If you skip it, MD simulations will still run but will use the CPU.

These commands should be executed from your home directory (`cd ~`) to keep sources separate from project files.

## 1. Clean up previous attempts (if necessary)

If you have old `openmm` or `openmm-metal` source folders in your home directory, remove them:

```bash
sudo rm -rf ~/openmm ~/openmm-metal
```

## 2. Create a version-pinned Conda environment

The OpenMM library and its source code must match exactly. Re-create the `prometheus` environment and pin OpenMM to 8.0.0, the version known to work with the Metal plugin:

```bash
conda deactivate          # if any env is active
mamba env remove -n prometheus

mamba create -n prometheus -c conda-forge python=3.10 rdkit smina openbabel "openmm=8.0.0" pdbfixer openmmforcefields
mamba activate prometheus
```

## 3. Install Python libraries

```bash
cd path/to/your/project-prometheus
pip install -r requirements.txt
```

## 4. Compile and install the OpenMM-Metal plugin

```bash
cd ~
git clone https://github.com/openmm/openmm
cd openmm
git checkout 8.0.0
cd ..

git clone https://github.com/philipturner/openmm-metal
cd openmm-metal

bash build.sh --install
```

If the command completes without errors your GPU-accelerated environment is ready. The `MDValidatorAgent` will automatically detect and use the new Metal platform.
