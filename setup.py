from setuptools import setup, find_packages

setup(
    name="ros2_auto_sysid",
    version="0.1.0",
    packages=find_packages(include=["sysid_core*", "sysid_models*", "sysid_excite*", "sysid_exporter*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "test": ["pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "run_sim2sim_benchmark = scripts.run_sim2sim_benchmark:main",
            "sysid_excitation_node = sysid_excite.excitation_node:main",
        ],
    },
    author="Robotics SysId Working Group",
    author_email="sysid@ros.org",
    description="Automated grey-box system identification and differentiable physics engine for ROS 2",
    license="Apache-2.0",
)
