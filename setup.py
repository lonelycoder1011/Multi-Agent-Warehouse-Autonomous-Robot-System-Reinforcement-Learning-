from setuptools import setup, find_packages

setup(
    name="warehouse-rl",
    version="1.0.0",
    description="Multi-Agent Autonomous Warehouse Reinforcement Learning System",
    author="WarehouseRL Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "ray[rllib]==2.10.0",
        "gymnasium==0.28.1",
        "torch==2.2.2",
        "numpy==1.26.4",
        "fastapi==0.110.2",
        "wandb==0.16.6",
    ],
    entry_points={
        "console_scripts": [
            "warehouse-train=training.trainer:main",
            "warehouse-eval=training.evaluate:main",
            "warehouse-dashboard=dashboard.app:main",
        ]
    },
)
