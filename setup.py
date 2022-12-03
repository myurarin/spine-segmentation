from setuptools import setup
from setuptools import find_packages

setup(name='spine-segmentation',
      version='0.0.1',
      description='背表紙分割',
      author='YuyaMatsuura',
      author_email='',
      url='',
      packages=find_packages("spine_segmentation"),
      install_requires=[
          "opencv-python"
      ]
      )
