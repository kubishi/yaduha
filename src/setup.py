from setuptools import setup, find_packages

setup(
    name='yaduha',
    version='0.1',
    packages=['yaduha'],
    install_requires=[
        'numpy>=2.0.0',
        'pandas==2.0.0',
        'python-dotenv==1.0.1',
        'openai==1.51.0',
        'spacy==3.8.2',
    ],
    dependency_links=[
        'git+ssh://git@github.com/kubishi/rbo.git#egg=rbo',
    ]
)