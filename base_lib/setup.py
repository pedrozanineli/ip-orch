from setuptools import setup, find_packages

setup(
    name='mlip',
    version='0.1.0',
    description='Use multiples UIPs models at the same time',
    author='Mauri Pereira dos Santos Júnior',
    author_email='maurisantosjr@gmail.com',
    url='https://github.com/maurijr1/mm_uip',
    packages=find_packages(),
    install_requires=["numpy"],  # Dependências
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

