from setuptools import setup, find_packages

setup(
    name='credit-risk-model',
    version='1.0.0',
    author='Your Name',  # CHANGE THIS
    author_email='your.email@example.com',  # CHANGE THIS
    description='Production-ready credit risk assessment model for financial institutions',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/credit-risk-model',  # CHANGE THIS
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas>=1.4.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'plotly>=5.10.0',
        'streamlit>=1.20.0',
        'shap>=0.41.0',
    ],
    entry_points={
        'console_scripts': [
            'credit-risk-train=main:main',
        ],
    },
    python_requires='>=3.9',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Financial and Insurance Industry',
    ],
)
