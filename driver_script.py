import subprocess

program_list = ['encode_faces.py', 'cluster_faces.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)