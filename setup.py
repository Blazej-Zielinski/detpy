import setuptools
import os
import io

current_path = os.path.dirname(os.path.realpath(__file__))

with io.open(f"{current_path}/README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="detpy",
    packages=setuptools.find_packages(),
    package_data={
        "detpy": ["functions/functions_info/*.json"],
    },
    version="1.0.11",
    author="Szymon Ściegienny, Błażej Zieliński, Hubert Orlicki, Wojciech Książek",
    author_email="wojciech.ksiazek@pk.edu.pl",
    description="DetPy (Differential Evolution Tools): A Python toolbox for solving optimization problems "
                "using differential evolution",
    keywords=["optimization", "metaheuristics", "nature-inspired algorithms",
              "evolutionary computation", "population-based algorithms",
              "Stochastic optimization", "different evolution"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Blazej-Zielinski/detpy",
    include_package_data=True,
    license="GPLv3",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Benchmark",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    install_requires=["numpy", "opfunu", "matplotlib", "tqdm", "scipy", "sympy", "autograd", "pandas"],
    python_requires='>=3.7',
)
