import numpy as np

def compute_trust(A,b,xdotj,xdotj_nominal,h,min_dist = 1.0,h_min = 1.0): # h > 0
    
    # distance
    rho_dist = A @ xdotj - b;
    try: 
        assert(rho_dist>0.001) # should always be positive
    except:
        print(f"Assertion failed rho_dist:{rho_dist}") 
        rho_dist = 0.001

    # if h < h_min:
    #     print(f"small h: {h}")
    rho_dist = np.tanh(0.01*rho_dist) #np.tanh(3.0*rho_dist); # score between 0 and 1  
    
    # angle
    if np.linalg.norm(xdotj)>0.01:
        theta_as = np.real(np.arccos( A @ xdotj/np.linalg.norm(A)/np.linalg.norm(xdotj) / 1.05))
    else:
        theta_as = np.arccos(0.001)
    if np.linalg.norm(xdotj_nominal)>0.01:
        theta_ns = np.real(np.arccos( A @ xdotj_nominal/np.linalg.norm(A)/np.linalg.norm(xdotj_nominal)/1.05 )) 
    else:
        theta_ns = np.arccos(0.001)
    # if (theta_ns<0.05):
    #     theta_ns = 0.05

    rho_theta = np.tanh(theta_ns/theta_as*0.9) #0.9##0.55 # if it is close to it's nominal, then trust high. if far away from it's nominal, then trust low     
    # print(f"rho_dist:{rho_dist}")
    if rho_dist>min_dist: # always positive
        trust = 3*rho_theta*rho_dist#(rho_dist-min_dist)
    else: # danger
        if h>h_min:  # far away. therefore still relax/positive
            trust = 3*rho_theta*rho_dist
        else:  # definitely negative this time
            trust = -2*(1-rho_theta)*rho_dist
            # print(f"Negative Trust! h: {h}, h_min:{h_min}")
            
            
    asserted = True
    try: 
        assert(rho_dist>0.001) # should always be positive
    except:
        # print(f"Assertion failed rho_dist:{rho_dist}, trust:{trust}") 
        asserted = False
        
    return trust, asserted

def compute_trust2(A,b,xdotj,xdotj_nominal,h,min_dist = 1.0,h_min = 1.0): # h > 0
    
    # distance
    rho_dist = A @ xdotj - b;
    # try: 
    #     assert(rho_dist>0.001) # should always be positive
    # except:
    #     print(f"Assertion failed rho_dist:{rho_dist}") 

    # if h < h_min:
    #     print(f"small h: {h}")
    rho_dist = np.tanh(0.01*rho_dist) #np.tanh(3.0*rho_dist); # score between 0 and 1  
    
    # angle
    if np.linalg.norm(xdotj)>0.01:
        theta_as = np.real(np.arccos( A @ xdotj/np.linalg.norm(A)/np.linalg.norm(xdotj) / 1.05))
    else:
        theta_as = np.arccos(0.001)
    if np.linalg.norm(xdotj_nominal)>0.01:
        theta_ns = np.real(np.arccos( A @ xdotj_nominal/np.linalg.norm(A)/np.linalg.norm(xdotj_nominal)/1.05 )) 
    else:
        theta_ns = np.arccos(0.001)
    # if (theta_ns<0.05):
    #     theta_ns = 0.05

    rho_theta = np.tanh(theta_ns/theta_as*0.9) #0.55 # if it is close to it's nominal, then trust high. if far away from it's nominal, then trust low     
    # print(f"rho_dist:{rho_dist}")
    if rho_dist>min_dist: 
        if h>h_min:  # far away. therefore still relax/positive
            trust = 3*rho_theta*rho_dist
        else:  # definitely negative this time # danger 
            print(f"Negative Trust! h: {h}, h_min:{h_min}")
            trust = -2*(1-rho_theta)*rho_dist
    else: # always positive
        trust = 3*rho_theta*rho_dist#(rho_dist-min_dist)
        
            
    asserted = True
    try: 
        assert(rho_dist>0.001) # should always be positive
    except:
        # print(f"Assertion failed rho_dist:{rho_dist}, trust:{trust}") 
        asserted = False
        
    return trust, asserted