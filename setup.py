from setuptools import find_packages, setup

setup(
    name="ares",
    version="0.1.0",
    description="ARES: Adaptive Reasoning Effort Selection for Efficient LLM Agents",
    license="MIT",
    packages=find_packages(include=["ares", "tau_bench", "tau_bench.*"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "litellm>=1.41.0",
        "openai>=1.13.3",
        "pydantic>=2",
        "transformers",
        "datasets",
        "tqdm",
    ],
    extras_require={
        "sft": ["torch", "accelerate", "deepspeed"],
        "rl": ["pyyaml"],
    },
)
