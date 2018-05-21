from setuptools import setup
from setuptools import find_packages


setup(name='jumpy',
      version='0.2.3',
      description='Numpy and nd4j interop',
      long_description='Mapping of the numpy & nd4j array representations',
      author='Adam Gibson',
      author_email='adam@skymind.io',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
      ],
      keywords='numpy jumpy java nd4j deeplearning4j',
      url='https://github.com/deeplearning4j/jumpy',
      license='Apache',
      setup_requires=['Cython'],
      install_requires=['Cython', 'pyjnius', 'numpy'],
      extras_require={
          'visualize': ['pydot-ng'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'mock'],
      },
      packages=find_packages())
