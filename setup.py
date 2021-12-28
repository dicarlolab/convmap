from distutils.core import setup

setup(
    name='CubeMap',
    version='0.1',
    packages=['convmap'],
    install_requires=['numpy', 'scipy', 'h5py', 'tensorflow-gpu'],
    url='https://github.com/dicarlolab/convmap.git',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Pouya Bashivan',
    description="Neural mapping from convolutional feature maps."
)
