import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

# INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setuptools.setup(
    name='vibe-lwgan',
    version='1.0.0',
    description='VIBE: Video Inference for Human Body Pose and Shape Estimation',
    author='mkocabas,vadim,anton',
    license='MIT License',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://holoanton@bitbucket.org/vadimkalikin/vibe.git',
    packages=setuptools.find_packages(),
    # install_requires=INSTALL_REQUIREMENTS,
    python_requires='>=3.6',
)