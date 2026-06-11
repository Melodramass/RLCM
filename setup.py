from setuptools import setup, find_packages

setup(
    name='rlcm',
    version='0.1.0',
    description='RLCM: Process Supervision of Confidence Margin for Calibrated LLM Reasoning',
    license='Apache License 2.0',
    packages=find_packages(include=['rlcm', 'rlcm.*']),
    install_requires=[
        'latex2sympy2',
        'pylatexenc',
        'sentence_transformers',
        'tabulate',
    ],
    python_requires='>=3.10',
)
