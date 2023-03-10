from setuptools import setup

with open("LIBRARY.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name='InterpretME',
    packages=['InterpretME'],
    version='1.2.1',
    description='An interpretable machine learning pipeline over knowledge graphs',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Yashraj Chudasama, Disha Purohit, Philipp Rohde, Julian Gercke',
    author_email='yashraj.chudasama@tib.eu',
    url='https://github.com/SDM-TIB/InterpretME',
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Operating System :: OS Independent",
        'License :: OSI Approved :: MIT License',
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.8, <3.10',
    install_requires=[
        'pandas>=1.4.1',
        'imbalanced-learn>=0.9.0',
        'lime>=0.2.0',
        'pydotplus>=2.0.2',
        'svglib>=1.2.1',
        'colour>=0.1.5',
        'matplotlib<=3.3.4',
        'rdflib<=6.1.1',
        'seaborn>=0.11.2',
        'numpy>=1.23.5',
        'dtreeviz>=1.3.0,<2.0.0',
        'python-slugify>=6.0.0',
        'requests>=2.27.0',
        'rdfizer>=4.5.5',
        'DeTrusty>=0.11.1',
        'validating-models>=0.9.0',
        'optuna>=3.1.0'
    ]
)
