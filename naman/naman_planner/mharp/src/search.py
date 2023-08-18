import heapq
import copy

class Search(object): 
    
    @staticmethod
    def gbfs(init, goal, actions, abstraction, monitor): 
        q = [ ]
        visited = set() 
        path = [init] 
        heapq.heappush(q,(0,(copy.deepcopy(init),copy.deepcopy(path),0)))
        while len(q) > 0: 
            current, path, depth = heapq.heappop(q)[1]
            if current == goal:
                return path
            visited.add(current)
            for dest_id in actions[current.id]:
                action = actions[current.id][dest_id] 
                if action.dest not in visited: 
                    h1 = abstraction.estimate_heuristic(action.dest, goal)
                    h2 = monitor.get_robot_dynamic_cost(action.dest, depth+1)
                    h3 = float(abstraction.get_abstract_state_volume(action.dest))
                    h_final = h1 + (h2 * 100000)/float(h3)
                    # print (h2 * 10000)/float(h3)
                    heapq.heappush(q,( h_final, (copy.deepcopy(action.dest), copy.deepcopy(path) + [action.dest], depth + 1 )))
        return []    