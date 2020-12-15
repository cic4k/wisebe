from io import open
from os import path
import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent
# The text of the README file
README = (HERE / "README.md").read_text()
# automatically captured required modules for install_requires in requirements.txt and as well as configure dependency links
with open(path.join(HERE, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if ('git+' not in x) and (not x.startswith('#')) and (not x.startswith('-'))]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if 'git+' not in x]

setup(
    name = 'wisebe',
    description = 'WiSeBE evaluation tool',
    version = '1.0.0',
    packages = find_packages(), # list of all packages
    install_requires = install_requires,
    python_requires='>=3.9', # any python greater than 2.7
    entry_points='''
        [console_scripts]
        wisebe=wisebe.__main__:main
    ''',
    author="Carlos González",
    keyword="sentence boundary detection evaluation",
    long_description=README,
    long_description_content_type="text/markdown",
    license='MIT',
    url='https://github.com/cic4k/wisebe',
    download_url='https://github.com/cic4k/wisebe/archive/1.0.0.tar.gz',
    dependency_links=dependency_links,
    author_email='carlos-emiliano.gonzalez-gallardo@sorbonne-universite.fr',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ]
)
