from setuptools import setup

import versioneer


def readme():
    with open('README.rst') as file_:
        return file_.read()


setup(name="gftools",
      version=versioneer.get_version(),
      description="FIXME",
      long_description=readme(),
      url="https://github.com/DerWeh/gftools",
      author="Weh",
      author_email="andreas.weh@student.uni-augsburg.de",
      cmdclass=versioneer.get_cmdclass(),
      packages=['gftools'],
      install_requires=[
          'numpy',
          'scipy',
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],)
