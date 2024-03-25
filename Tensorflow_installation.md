# Tensorflow installation

This guide will walk you through the installation of Tensorflow 2.15.0.

## Step 1: Create a Conda Environment

First, we need to create a new conda environment. You can do this using the following command:

```bash
conda create --name tf python=3.11 -y
```

Where "tf" is the name of the environment. The Python version has to match one of the versions supported by the tensorflow version to be installed, so it is best to [visit the Tensorflow web page](https://www.tensorflow.org/install/source).

## Step 2: Activate the Environment

Next, activate the newly created environment:

```bash
conda activate tf
```

## Step 3: Install PIP

Now, we need to install PIP in our environment:

```bash
conda install pip -y
```

## Step 4: Upgrade PIP

It's a good practice to upgrade PIP to the latest version:

```bash
pip install --upgrade pip
```

## Step 5: Install CUDA Toolkit

We will install the CUDA Toolkit from the Nvidia CUDA 12.2.2 label:

```bash
conda install -c "nvidia/label/cuda-12.2.2" cuda-toolkit -y
python3 -m pip install nvidia-cudnn-cu11==8.9.6.50
```

As with the Python version, you must install the CUDA version compatible with the Tensorflow version to be installed.

## Step 6: Install Tensorflow

Now, we can install Tensorflow:

```bash
pip install tensorflow==2.15.0.post1
```

## Step 7: Test

Now we can open the preferred code editor, in my case I used Visual Studio Code so we also have to install the Python and Jupyter extensions, to be able to use the notebooks. We create a new notebook and use the following code, if it appears the number of GPUs we have, the process will have worked correctly:

```python
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
```