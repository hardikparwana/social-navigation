import time
from scipy.spatial import distance
import numpy as np
from heapq import *

class HARP(object):

    def __init__(self,env, start,goal, abstraction, discretizer, abstract_plan_state_ids, high_level_state_sampler,  critical_regions_sampler, visualize = False, n = 250, m = 500, max_time = 200):
        self.start = start[:2]
        # self.start = discretizer.convert_sample(start[:2])
        # self.goal = discretizer.convert_sample(goal[:2])
        self.goal = goal[:2]
        self.start_theta = start[-1]
        self.goal_theta = goal[-1]
        self.critical_regions = critical_regions_sampler
        self.sampling_distribution = high_level_state_sampler
        self.abstraction = abstraction
        self.n = n 
        self.m = m
        self.roadmap = None
        self.roadmapedges = None
        self.selected_graph = 0
        self.goal_radius = 0.20
        self.max_time = max_time
        self.starttime = time.time()
        self.currentgraph = 0
        self.currentstate = "build graphs"
        self.env = env
        self.hl_regions = abstract_plan_state_ids
        self.k = 0
        self.discretizer = discretizer
        self.llimits = self.discretizer.robot.get_dof_lower_limits()
        self.ulimits = self.discretizer.robot.get_dof_upper_limits()
        self.visualize = visualize
        self.trace  = []
        if self.visualize:
            self.trace.append(self.env.plot3(points=[self.start[0], self.start[1], 1.5], pointsize=0.07, colors=np.array([0, 1, 0]), drawstyle=1))
            self.trace.append(self.env.plot3(points=[self.goal[0], self.goal[1], 1.5], pointsize=0.07, colors=np.array([0, 0, 1]), drawstyle=1))
        self.build_roadmap()



    def normalize(self,q):
        normalized = []
        for i in range(len(q)): 
            normalized.append((1.0 / float(self.ulimits[i] - self.llimits[i])) * (q[i] - self.ulimits[i]) + 1 )
        return normalized

    def normalized_dist(self,a,b,flag=True):
        if flag:
            n_a = self.normalize(a)
        else:
            n_a = a
        n_b = self.normalize(b)
        return np.linalg.norm(np.asarray(n_a) - np.asarray(n_b))

    def sample_critical_regions(self):
        return self.critical_regions.sample()
        
    def sample_abstract_states(self):
        #TODO: implement high-level state sampling to reduce the state space.
        return self.sampling_distribution.sample()

    def collides(self,q):
        collision = False
        robot = self.env.GetRobots()[0]
        with self.env: 
            prev  = robot.GetActiveDOFValues() 
            robot.SetActiveDOFValues(q)
            if self.env.CheckCollision(robot) or robot.CheckSelfCollision(): 
                collision = True
            robot.SetActiveDOFValues(prev)
        return collision
              
    def get_new_sample(self):
        if self.k < 300:
            self.k += 1
            return self.sample_critical_regions()
        else:
            return self.sample_abstract_states()

    def build_roadmap(self):
        self.roadmap = []
        self.roadmapedges = []

        while len(self.roadmap) < self.n:
            while True:
                q = self.sample_critical_regions()
                if not self.collides(q):
                    # self.simulator.env.plot(q,[0,0,1])
                    break
                else:
                    # self.simulator.env.plot(q,[1,0,0])
                    pass
            self.roadmap.append([q])
            self.roadmapedges.append({0:[]})
            
        while (len(self.roadmap)-self.n) < self.m:
            while True:
                q = self.sample_abstract_states()
                if not self.collides(q):
                    break
            self.roadmap.append([q])
            self.roadmapedges.append({0:[]})


        if len(self.start) > 0 and len(self.goal) > 0:
            self.roadmap.append([self.start])
            self.roadmapedges.append({0:[]})
            self.roadmap.append([self.goal])
            self.roadmapedges.append({0:[]})

        self.buildtime = self.max_time

        while True:
            if (time.time()-self.starttime) <= self.buildtime: # 1 seconds to build in RM mode. 60 in llp
                rand = self.get_new_sample()
                # self.simulator.env.plot(rand,[0,0,1])
                self.roadmap[self.currentgraph], self.roadmapedges[self.currentgraph], status, new = self.extend(self.roadmap[self.currentgraph], self.roadmapedges[self.currentgraph], rand)

                if status != 'trapped':
                    connected = self.connectN(new)

                    if connected:
                        # print str(time.time()-self.starttime) + ',' + str(len(self.roadmap[0])) # comment put for llp
                        self.currentstate = 'connected graphs'
                        break
                    pass

                if self.currentstate == 'build graphs':
                    self.swapN()
            else:
                print("Time up", str(time.time()-self.starttime))
                self.currentstate = 'connected graphs'
                import IPython
                IPython.embed()
                numstates = 0
                for m in self.roadmap:
                    numstates += len(m)
                break
    
    def swapN(self):
        if self.currentgraph >= len(self.roadmap)-1:
            self.currentgraph = 0
        else:
            self.currentgraph += 1

    def connect(self, V, E, q):
        status = 'advanced'

        # loop until reached or collision
        while status == 'advanced':
            V, E, status, new = self.extend(V, E, q)

        if status == 'reached':
            # add G=(V,E) to q's graph
            i_q = len(self.roadmap[self.currentgraph])-1 
            self.roadmap[self.currentgraph] = self.roadmap[self.currentgraph] + V
            for i, e in enumerate(E):
                adj = []
                for n in E[e]:
                    adj.append((n[0]+i_q+1,n[1]))
                self.roadmapedges[self.currentgraph][i_q+1+i] = adj

            i_new = len(self.roadmap[self.currentgraph])-1
            self.roadmapedges[self.currentgraph][i_q].append((i_new,new))
            self.roadmapedges[self.currentgraph][i_new].append((i_q,q))
            # self.trace.append(self.env.drawlinestrip(points=array([[q[0], q[1], self.traceheight],[new[0], new[1], self.traceheight]]), linewidth=0.5, colors=array(self.color), drawstyle=1)
        return status 

    def connectN(self, q):
        delete = []
        for i in range(len(self.roadmap)):
            if i != self.currentgraph:
                connected = self.connect(self.roadmap[i], self.roadmapedges[i], q)
                if connected == 'reached': 
                    delete.append(i)
            if (time.time()-self.starttime) >= self.buildtime:
                break

        # delete merged graphs
        self.roadmap = [self.roadmap[i] for i in range(len(self.roadmap)) if not i in delete]
        self.roadmapedges = [self.roadmapedges[i] for i in range(len(self.roadmapedges)) if not i in delete]
        # print len(self.roadmap)
        # if self.constrained and len(self.roadmap) < 50:
        if self.check_if_connected():
                return True

        return len(self.roadmap) == 1

    def check_if_connected(self):
        for i in range(len(self.roadmap)):
            if self.start in self.roadmap[i] and self.goal in self.roadmap[i]:
                self.selected_graph = i
                return True
        return False

    def extend(self, V, E, q):
        try:
            # i_near = distance.cdist([q], V,self.normalized_dist).argmin()
            i_near = distance.cdist([q], V).argmin()
        except ValueError:
            pass
        near = V[i_near]
        new = self.compound_step(near, q)
        if not self.abstraction.check_env_collision(new) and self.collides(new) == False and self.abstraction.get_abstract_state(new).id in self.hl_regions:  
            V.append(new)
            E[len(V)-1] = []
            E[len(V)-1].append((i_near,near)) 
            E[i_near].append((len(V)-1,new))
            if self.visualize:
                self.trace.append(self.env.plot3(points=[near[0], near[1], 1.5], pointsize=0.01, colors=np.array([1, 0, 0]), drawstyle=1))
                self.trace.append(self.env.drawlinestrip(points=np.array([[near[0], near[1], 1.5], [new[0], new[1], 1.5]]), linewidth=0.5,colors=np.array([1,0,0]), drawstyle=1))
            if self.goal_zone_collision(new, q):
                return V, E, 'reached', new
            else:
                return V, E, 'advanced', new
        else:
            return V, E, 'trapped', None 

    def compound_step(self,p1,p2):
        a = []
        for i in range(len(p1)):
            a = a + self.step_from_to([p1[i]],[p2[i]],0.15)
        return a

    def step_from_to(self,p1,p2,distance):
        #https://github.com/motion-planning/rrt-algorithms/blob/master/src/rrt/rrt_base.py
        if self.dist(p1,p2) <= distance:
        # if self.normalized_dist(p1,p2) <= distance:
            return p2
        else:
            a = np.array(p1)
            b = np.array(p2)
            ab = b-a  # difference between start and goal

            zero_vector = np.zeros(len(ab))

            ba_length = self.dist(zero_vector, ab)  # get length of vector ab
            unit_vector = np.fromiter((i / ba_length for i in ab), np.float, len(ab))
            # scale vector to desired length
            scaled_vector = np.fromiter((i * distance for i in unit_vector), np.float, len(unit_vector))
            steered_point = np.add(a, scaled_vector)  # add scaled vector to starting location for final point

            return list(steered_point)
    
    def goal_zone_collision(self,p1,p2):
        # if self.normalized_dist(p1, p2) <= self.goal_radius:
        if self.dist(p1, p2) <= self.goal_radius:
            return True
        else:
            return False

    def dist(self,a,b):
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a-b)


    def search(self):
        ''' dijkstra's '''
        q = []
        dist = {}
        prev = {}
        
        for i in range(len(self.roadmap[self.selected_graph])):
            dist[i] = float("inf")
            prev[i] = None

        dist[self.start[0]] = 0
        heappush(q, (0,self.start))

        while q:
            currdist, near = heappop(q)

            for n in self.roadmapedges[self.selected_graph][near[0]]:
                # alt = currdist + self.dist(near[1], n[1])
                alt = currdist + self.dist(near[1], n[1])
                if alt < dist[n[0]]:
                    dist[n[0]] = alt
                    prev[n[0]] = near
                    heappush(q, (alt, n))

        # collect solution path through backtracking from goal using prev
        solutiontrace = []
        temp = self.goal
        if prev[temp[0]]:
            while temp:
                solutiontrace.append(temp[1])
                temp = prev[temp[0]]

        return solutiontrace

    def smooth(self,path, threshold = 0.3):
        smoothened = []
        i = 0 
        while i < len(path) - 1:
            p1 = path[i]
            last_reachable_state = i
            j = i + 1 
            while j < len(path): 
                p2 = path[j]
                # if self.dist(p1,p2) <= threshold:
                if self.dist(p1,p2) <= threshold:
                    last_reachable_state = j
                j+=1
            smoothened.append(path[last_reachable_state])
            if i == last_reachable_state:
                i += 1
            else:
                i = last_reachable_state
        return smoothened
    
    def compute_theta(self,path):
        new_path = [ path[0]+[self.start_theta] ]
        current1 = new_path[0]
        current2 = path[0]
        for next_wp in  path[1:]: 
            relative = np.asarray(next_wp) - np.asarray(current2)
            theta = np.arctan2(relative[1],relative[0])
            if not np.allclose(theta,current1[-1]): 
                new_path.append(current2 + [theta])
            new_path.append(list(next_wp) + [theta])
            current2 = next_wp 
            current1 = new_path[-1]
        new_path.append(list(path[-1]) + [self.goal_theta])
        return new_path
             

    def get_mp(self):
        s = self.start 
        g = self.goal
        if self.currentstate == 'connected graphs':
            i_s = distance.cdist([s], self.roadmap[self.selected_graph]).argmin()
            i_g = distance.cdist([g], self.roadmap[self.selected_graph]).argmin()
            self.start = (i_s,s)
            self.goal = (i_g,g)

            path = self.search()
            if len(path) > 0:
                path = self.compute_theta(self.smooth(path[::-1]))
                return True, path
            else:
                return False, []
        else:
            return False, []