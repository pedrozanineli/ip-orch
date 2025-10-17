import subprocess

def set_calc(uip, script_path):
	activate_cmd = (
		f"source $(conda info --base)/etc/profile.d/conda.sh && "
		f"conda activate {uip} && "
		f"python {script_path} {uip}"
	)
	subprocess.run(activate_cmd, shell=True, executable="/bin/bash")	

