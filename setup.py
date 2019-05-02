from setuptools import setup
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rl_environments'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'abstract_classes'))

setup(name='rl_environments',
      packages=['rl_environments', 'rl_environments.vrep', 'rl_environments.pybullet', 'abstract_classes'],
      version='0.0.2',
      install_requires=['pybullet', 'vrep_env'],
      package_data={'': ['*.xml','*.urdf']},
      include_package_data=True
)