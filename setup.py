from setuptools import setup
from setuptools import find_packages

install_requires = [
    'tensorflow==0.5.0'
]

setup(
      name='Project_RNN_Enhancement',
      version='0.0.1',
      description='Additional RNN and Seq2Seq Features for TensorFlow',
      author='LeavesBreathe',
      url='https://github.com/LeavesBreathe/Project_RNN_Enhancement',
      license='Apache v2',
      install_requires=install_requires,
      packages=find_packages()
)