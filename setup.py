from setuptools import find_packages, setup
from typing import List

HYPHON_e_DOT = '-e .'
def get_requirements(file_path:str)-> List[str]:
      '''
      This function will return the list of all the requirements.
      '''
      requirements = []
      with open(file_path) as obj:
            requirements = obj.readlines()
            requirements = [i.replace("\n", "") for i in requirements]
            
            if HYPHON_e_DOT in requirements:
                  requirements.remove(HYPHON_e_DOT)
            
            return requirements



setup(
      name = 'Customer_Churn',
      version = '0.0.1',
      author = 'Nihal',
      author_email = 'nihalk2180@outlook.com',
      packages = find_packages(),
      install_requires = get_requirements('requirements.txt')
)