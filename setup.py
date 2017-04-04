from setuptools import setup
from setuptools import find_packages

setup(name='jumpy',
      version='0.1',
      description='Numpy and nd4j interop',
      author='Adam Gibson',
      author_email='adam@skymind.io',
      url='https://github.com/deeplearning4j/jumpy',
      download_url='https://github.com/fchollet/keras/tarball/2.0.2',
      license='MIT',
      install_requires=['six', 'pyjnius'],
      extras_require={
          'visualize': ['pydot-ng'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist'],
      },
      packages=find_packages())