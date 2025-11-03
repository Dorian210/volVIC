from setuptools import setup, find_packages

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(
    name='volVIC', 
    version='1.0.0', 
    url='https://github.com/Dorian210/volVIC', 
    author='Dorian Bichet', 
    author_email='dbichet@insa-toulouse.fr', 
    description='', 
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    packages=find_packages(), 
    install_requires=['numpy', 'numba', 'scipy', 'matplotlib', 'meshio', 'tqdm', 'scikit-sparse', 'pyvista', 'bsplyne', 'interpylate', 'treeIDW'], 
    classifiers=['Programming Language :: Python :: 3', 
                 'Operating System :: OS Independent', 
                 'License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)'], 
    
)
