"""

Extension of SIPP to multi-robot scenarios

author: Ashwin Bose (@atb033)

See the article: 10.1109/ICRA.2011.5980306

"""

import argparse
import yaml
from math import fabs
from graph_generation import SippGraph, State
from sipp import SippPlanner
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map", help="input file containing map and dynamic obstacles")
    parser.add_argument("output", help="output file with the schedule")
    parser.add_argument("seed", help = "seed", default=10, type=int)

    
    args = parser.parse_args()
    np.random.seed(args.seed)
    # Read Map
    with open(args.map, 'r') as map_file:
        try:
            map = yaml.load(map_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    # Output file
    output = dict()
    output["schedule"] = dict()
    nums = list(range(len(map["agents"])))

    for i in range(len(map["agents"])):
        j = np.random.choice(nums)
        sipp_planner = SippPlanner(map,j)
        nums.remove(j)
    
        if sipp_planner.compute_plan():
            plan = sipp_planner.get_plan()
            output["schedule"].update(plan)
            map["dynamic_obstacles"].update(plan)

            with open(args.output, 'w') as output_yaml:
                yaml.safe_dump(output, output_yaml)  
        else: 
            plan = sipp_planner.get_plan(failure=True)
            output["schedule"].update(plan)
            map["dynamic_obstacles"].update(plan)

            with open(args.output, 'w') as output_yaml:
                yaml.safe_dump(output, output_yaml)  


if __name__ == "__main__":
    main()
