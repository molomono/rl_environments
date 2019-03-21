from setuptools import setup
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rl_environments'))

setup(name='rl_environments',
      version='0.0.1',
      install_requires=['gym',
                        'pybullet',
						'rl_coach']
)