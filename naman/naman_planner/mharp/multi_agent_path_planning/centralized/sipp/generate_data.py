import subprocess
import sys
import tqdm

yaml_folder = sys.argv[1]
output_folder = sys.argv[2]
n_problems = int(sys.argv[3])
n_mps = int(sys.argv[4])


for i in tqdm.tqdm(range(n_problems)): 
    problem_path = yaml_folder + "problem_{}.yaml".format(i)
    for j in range(n_mps):
        output_path = output_folder + "output_{}_{}.yaml".format(i,j)
        pid = subprocess.Popen(args=["python", "multi_sipp.py", problem_path, output_path,  str(j) ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pid.wait()
        pid.kill()