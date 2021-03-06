from setuptools import setup
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rl_environments'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'abstract_classes'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

setup(name='rl_environments',
      packages=['rl_environments', 'rl_environments.vrep', 'rl_environments.pybullet'],
      version='0.1.0',
      install_requires=['pybullet', 'vrep_env'],
      package_data={'': ['*.xml','*.urdf', '*.json', '*.yaml', '*.pickle']},
      include_package_data=True
)