import subprocess

with open("README.md.template") as fh:
    template = fh.read()

output = subprocess.check_output("scip --help", shell=True).decode("utf-8")
with open("README.md", "w") as fh:
    fh.write(template.format(usage=output))
