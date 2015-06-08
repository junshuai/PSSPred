from setuptools import setup

setup(
    name='psspred',
    version='0.1',
    py_modules=['psspred'],
    include_package_data=True,
    install_requires=[
        'click',
        'numpy',
        'theano',
    ],
    entry_points='''
        [console_scripts]
        psspred=psspred:cli
    ''',
)
