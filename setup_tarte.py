import os
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Create env.common. Optionally install pip dependencies."
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install pip dependencies from $PROJECT_ROOT/requirements/requirements_tabicl_tabpfn.txt",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Detect active conda environment
    # ------------------------------------------------------------------
    CONDA_ENV_PATH = os.environ.get("CONDA_PREFIX")
    if not CONDA_ENV_PATH:
        raise RuntimeError("No active conda environment detected. Run 'conda activate <env>' first.")

    # ------------------------------------------------------------------
    # Detect project root (current working directory)
    # ------------------------------------------------------------------
    PROJECT_ROOT = os.getcwd()

    # ------------------------------------------------------------------
    # Create env.common
    # ------------------------------------------------------------------
    env_file = Path("env.common")

    with env_file.open("w") as f:
        f.write("# Auto-generated environment variables\n")
        f.write(f"export CONDA_ENV_PATH='{CONDA_ENV_PATH}'\n")
        f.write(f"export PROJECT_ROOT='{PROJECT_ROOT}'\n")

    print(f"[INFO] Created {env_file}")
    print(f"       CONDA_ENV_PATH={CONDA_ENV_PATH}")
    print(f"       PROJECT_ROOT={PROJECT_ROOT}")

    # ------------------------------------------------------------------
    # Optional Tarte dependency installation
    # ------------------------------------------------------------------
    if args.install_deps:
        REQUIREMENTS_PATH = Path(PROJECT_ROOT) / "requirements" / "requirements_tarte.txt"

        if not REQUIREMENTS_PATH.exists():
            raise FileNotFoundError(f"Requirements file not found: {REQUIREMENTS_PATH}")

        print(f"[INFO] Installing dependencies from {REQUIREMENTS_PATH}")
        subprocess.run(["pip", "install", "-r", str(REQUIREMENTS_PATH)], check=True)
        print("[INFO] Dependency installation completed successfully.")
    else:
        print("[INFO] Skipping dependency installation (use --install-deps to enable).")


if __name__ == "__main__":
    main()