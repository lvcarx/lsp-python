from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='lsp-python',
    version='0.0.3.post1',
    description='lsp-python is a lightweight implementation of the Least Square Projection (LSP) dimensionality reduction technique using a sklearn style API.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lvcarx/pyLSP',
    author='Luca Reichmann',
    author_email='st169765@stud.uni-stuttgart.de',
    license='MIT',
    packages=['lsp_python'],
    install_requires=['numpy',
                      'scikit-learn',
                      'matplotlib'],
    classifiers=[
        'Programming Language :: Python :: 3.11',
    ],
)
