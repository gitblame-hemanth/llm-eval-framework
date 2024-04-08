"""Package setup for llm-eval-framework."""

from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).resolve().parent

long_description = ""
readme = here / "README.md"
if readme.exists():
    long_description = readme.read_text(encoding="utf-8")

requirements = []
req_file = here / "requirements.txt"
if req_file.exists():
    requirements = [
        line.strip()
        for line in req_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="llm-eval-framework",
    version="0.1.0",
    description="Systematically evaluate LLM outputs against configurable test suites",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hemanth V",
    author_email="hemanth199820@gmail.com",
    license="MIT",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "llm-eval=src.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
