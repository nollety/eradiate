import os
from textwrap import dedent


def main():
    root_dir = os.getcwd()
    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_default_env = os.environ.get("CONDA_DEFAULT_ENV")

    # Check if conda env is active
    if not conda_default_env or conda_default_env == "base":
        print("Activate a Conda environment other than 'base'")
        exit(1)

    # Define paths to (de)activate scripts
    activate_dir = os.path.join(conda_prefix, "etc/conda/activate.d")
    deactivate_dir = os.path.join(conda_prefix, "etc/conda/deactivate.d")
    print("Copying environment variable setup scripts to:\n")
    print(f"    {activate_dir}")
    print(f"    {deactivate_dir}\n")

    # Copy environment setup scripts
    os.makedirs(activate_dir, exist_ok=True)
    with open(os.path.join(activate_dir, "env_vars.sh"), "w") as f:
        f.write(activate_script(root_dir))

    os.makedirs(deactivate_dir, exist_ok=True)
    with open(os.path.join(deactivate_dir, "env_vars.sh"), "w") as f:
        f.write(deactivate_script())

    # fmt: off
    print(dedent(f"""
    You might want to reactivate your environment:

        $ conda deactivate && conda activate {conda_default_env}
    """).lstrip())
    # fmt: on


def activate_script(eradiate_dir):
    # fmt: off
    return dedent(f"""
        #!/bin/sh
        ERADIATE_DIR="{eradiate_dir}"
        source ${{ERADIATE_DIR}}/setpath.sh
    """).lstrip()
    # fmt: on


def deactivate_script():
    # fmt: off
    return dedent(f"""
        #!/bin/sh
        # Clean up PATH
        match="${{ERADIATE_DIR}}/build/dist:"
        export PATH=${{PATH//$match/}}
        # Clean up PYTHONPATH
        match="${{ERADIATE_DIR}}/build/dist/python:"
        export PYTHONPATH="${{PYTHONPATH//$match/}}"
        # Remove other environment variables
        unset ERADIATE_DIR MITSUBA_DIR
    """).lstrip()
    # fmt: on


if __name__ == "__main__":
    main()