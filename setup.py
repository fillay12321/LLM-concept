from setuptools import find_packages, setup

setup(
    name="triad_llm",
    version="0.0.0",
    description="TRIAD-LLM-concept: Minkowski spacetime attention",
    packages=find_packages(),
    install_requires=["torch", "tiktoken"],
)
