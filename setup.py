# https://docs.vllm.ai/en/stable/contributing/model/registration.html
# https://docs.vllm.ai/en/stable/design/plugin_system.html

from setuptools import setup

setup(name='vllm_mbart',
      version='0.1',
      packages=['vllm_mbart'],
      install_requires=["vllm>=0.9.0"],
      entry_points={
          'vllm.general_plugins':
              ["register_mbart = vllm_mbart:register"]
      })
