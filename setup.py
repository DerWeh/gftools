from setuptools import setup

import versioneer


def readme():
    with open('README.rst') as file_:
        return file_.read()


setup(
    name="gftools",
    version=versioneer.get_version(),
    description="Collection of commonly used Green's functions and utilities",
    long_description=readme(),
    keywords=r"Green's\ function physics",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    url="https://github.com/DerWeh/gftools",
    project_urls={
        "Documentation": "https://derweh.github.io/gftools/",
        "ReadTheDocs": "https://gftools.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/DerWeh/gftools",
    },
    author="Weh",
    author_email="andreas.weh@student.uni-augsburg.de",
    cmdclass=versioneer.get_cmdclass(),
    packages=['gftools'],
    install_requires=[
        'numpy',
        'scipy',
        'mpmath'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'hypothesis'],
)
