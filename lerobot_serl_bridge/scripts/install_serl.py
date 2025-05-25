#!/usr/bin/env python3
"""
SERL Installation Script

This script helps install SERL dependencies and set up the integration
with LeRobot.
"""

import subprocess
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, check=True):
    """Run a shell command and return the result"""
    logger.info(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, check=check,
            capture_output=True, text=True
        )
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stderr:
            logger.error(f"Error: {e.stderr.strip()}")
        raise


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        raise RuntimeError(
            f"Python 3.8+ required, got {version.major}.{version.minor}")
    logger.info(
        f"Python version: {version.major}.{version.minor}.{version.micro}")


def install_jax():
    """Install JAX with appropriate backend"""
    logger.info("Installing JAX...")

    # Check if CUDA is available
    try:
        result = run_command("nvidia-smi", check=False)
        if result.returncode == 0:
            logger.info("CUDA detected, installing JAX with CUDA support")
            run_command(
                "pip install jax[cuda] -f "
                "https://storage.googleapis.com/jax-releases/"
                "jax_cuda_releases.html")
        else:
            logger.info("No CUDA detected, installing CPU-only JAX")
            run_command("pip install jax jaxlib")
    except Exception:
        logger.info("Installing CPU-only JAX")
        run_command("pip install jax jaxlib")


def install_serl():
    """Install SERL from the serl-main directory"""
    logger.info("Installing SERL...")

    # Find the serl-main directory
    current_dir = Path(__file__).parent.parent.parent
    serl_path = current_dir / "serl-main"

    if not serl_path.exists():
        logger.error(f"SERL directory not found at {serl_path}")
        logger.info("Please ensure serl-main is in the same parent directory")
        return False

    # Install SERL launcher
    serl_launcher_path = serl_path / "serl_launcher"
    if serl_launcher_path.exists():
        logger.info(f"Installing SERL launcher from {serl_launcher_path}")
        run_command(f"pip install -e {serl_launcher_path}")
    else:
        logger.error(f"SERL launcher not found at {serl_launcher_path}")
        return False

    # Install SERL robot infrastructure if available
    serl_robot_path = serl_path / "serl_robot_infra"
    if serl_robot_path.exists():
        logger.info(f"Installing SERL robot infra from {serl_robot_path}")
        run_command(f"pip install -e {serl_robot_path}")

    return True


def install_dependencies():
    """Install other required dependencies"""
    logger.info("Installing other dependencies...")

    dependencies = [
        "flax",
        "optax",
        "chex",
        "distrax",
        "gymnasium",
        "wandb",
        "tensorboard",
        "agentlace",
    ]

    for dep in dependencies:
        try:
            run_command(f"pip install {dep}")
        except subprocess.CalledProcessError:
            logger.warning(f"Failed to install {dep}, continuing...")


def verify_installation():
    """Verify that all components are installed correctly"""
    logger.info("Verifying installation...")

    # Test JAX
    try:
        import jax
        logger.info(f"JAX version: {jax.__version__}")
        logger.info(f"JAX devices: {jax.devices()}")
    except ImportError as e:
        logger.error(f"JAX import failed: {e}")
        return False

    # Test SERL
    try:
        from serl_launcher.agents.continuous.sac import SACAgent  # noqa: F401
        logger.info("SERL import successful")
    except ImportError as e:
        logger.error(f"SERL import failed: {e}")
        return False

    # Test bridge
    try:
        from lerobot_serl_bridge.training.online_trainer import (  # noqa: F401
            OnlineRLTrainer
        )
        logger.info("Bridge import successful")
    except ImportError as e:
        logger.error(f"Bridge import failed: {e}")
        return False

    logger.info("All components verified successfully!")
    return True


def main():
    """Main installation process"""
    logger.info("Starting SERL installation...")

    try:
        # Check Python version
        check_python_version()

        # Install JAX
        install_jax()

        # Install SERL
        if not install_serl():
            logger.error("SERL installation failed")
            return 1

        # Install other dependencies
        install_dependencies()

        # Verify installation
        if not verify_installation():
            logger.error("Installation verification failed")
            return 1

        logger.info("Installation completed successfully!")
        logger.info(
            "You can now run the example: "
            "python examples/serl_integration_example.py")

        return 0

    except Exception as e:
        logger.error(f"Installation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
