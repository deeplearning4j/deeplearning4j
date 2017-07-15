from setuptools import setup
from setuptools import find_packages


setup(name='jumpy',
      version='0.2.1',
      description='Numpy and nd4j interop',
      author='Adam Gibson',
      author_email='adam@skymind.io',
      url='https://github.com/deeplearning4j/jumpy',
      license='Apache',
      install_requires=['six', 'pyjnius'],
      extras_require={
          'visualize': ['pydot-ng'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist'],
      },
      packages=find_packages())