import sys, os, platform
if platform.system().startswith('L'):   # Means code is running in linux
    sys.path.append('/usr/share/sumo/tools')
    sumoBinary = "/usr/bin/sumo"

import os
import sys
import optparse
import traci
import xml.dom.minidom as minidom
import xml.etree.ElementTree as xee
import numpy as np
from sumolib import checkBinary


def Duration(info, action):
    """Return the time matrix based on info file"""
    total_time = np.zeros(action.shape)
    avg_time = np.zeros(action.shape)
    row, column = action.shape
    dom = minidom.parse(info)
    root = dom.documentElement
    trapinfos = root.getElementsByTagName('tripinfo')
    cnt = np.zeros(action.shape)
    for tripinfo in trapinfos:
        if tripinfo.getAttribute('vType') == 'CarA':
            begin_node=int(tripinfo.getAttribute('id')[1])
            end_node=int(tripinfo.getAttribute('id')[2])
            total_time[begin_node][end_node] += float(tripinfo.getAttribute('duration'))
            cnt[begin_node][end_node] += 1
    for i in range(row):
        for j in range(column):
            if action[i][j] != 0:
                avg_time[i][j] = total_time[i][j]/cnt[i][j]
    return avg_time


class SumoEnv:
    """Environment of SUMO used to simulate"""
    def __init__(self, setting, sim_steps):
        self.background = setting['EDGE_TRAFFIC']
        self.sim_steps = sim_steps
        self.path = os.getcwd()
        if self.path[-1]!= '\\' and self.path[-1]!='/':
            self.path += '/'
        if self.path[-6:-1] != 'model':    # If current path is .../gnn, then add the directory path
            self.path += 'transition_model/'

    def simulate(self, action, is_display = False):
        """Used to simulate and return the experienced traffic time
        params:
            action: [num_node, num_node] (np.ndarray) representing the proportion of idle drivers
            is_display: Bool, whether show sumo display window
        returns: a matrix [num_node, num_node] representing the simulated travelling time with [·]_{ij} if from node i to node j
        """
        self.action = action
        dom = minidom.getDOMImplementation().createDocument(None, 'routes', None)
        routes = dom.documentElement
        vType_taxi = dom.createElement('vType')
        vType_taxi.setAttribute("id","CarA")
        vType_taxi.setAttribute("length","5")
        vType_taxi.setAttribute("maxSpeed","22")
        vType_taxi.setAttribute("color","0,1,0")
        vType_background = dom.createElement('vType')
        vType_background.setAttribute("id","CarB")
        vType_background.setAttribute("length","5")
        vType_background.setAttribute("maxSpeed","22")
        vType_background.setAttribute("color","1,0,0")
        routes.appendChild(vType_taxi)
        routes.appendChild(vType_background)

        row, column = self.action.shape
        for i in range(row):
            for j in range(column):
                if action[i][j]!=0:
                    """ Set taxi flow"""
                    flow = dom.createElement('flow')
                    route = dom.createElement('route')
                    flow.setAttribute("id","A{0}{1}".format(i,j))
                    flow.setAttribute("type","CarA")
                    flow.setAttribute("begin","0")
                    flow.setAttribute("end","600")
                    flow.setAttribute("number",str(self.action[i][j]))
                    route.setAttribute("edges", "E{0}{1}".format(i,j))
                    routes.appendChild(flow)
                    flow.appendChild(route)
                    """ Set background flow"""
                    flow_b = dom.createElement('flow')
                    route_b = dom.createElement('route')
                    flow_b.setAttribute("id","B{0}{1}".format(i,j))
                    flow_b.setAttribute("type","CarB")
                    flow_b.setAttribute("begin","0")
                    flow_b.setAttribute("end","600")
                    flow_b.setAttribute("number",str(self.background[i][j]))
                    route_b.setAttribute("edges", "E{0}{1}".format(i,j))
                    routes.appendChild(flow_b)
                    flow_b.appendChild(route_b)
        # save rou.xml file based on current action
        with open(self.path+'info/current.rou.xml', 'w', encoding='utf-8') as f:
            dom.writexml(f, addindent='\t', newl='\n', encoding='utf-8')

        # save sumocfg file based on the new rou.xml file
        domTree = xee.parse(self.path + 'info/hello.sumocfg')
        # 获得所有节点内容
        root = domTree.getroot()
        # 获得所有标签是"input"的节点内容
        inputs = root.findall("input")
        input = inputs[0]
        route = input.findall("route-files")
        route[0].set("value", "current.rou.xml")
        domTree.write(self.path+"info/current.sumocfg", encoding="utf8")

        if is_display:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')

        traci.start([sumoBinary, "-c", self.path+"info/current.sumocfg"])
        for step in range(self.sim_steps):
            traci.simulationStep()
        traci.close()
        output = Duration(self.path+'info/info.xml', self.action)
        return output


if __name__ == "__main__":
    if 'SUMO_HOME' not in os.environ and platform.system().startswith("L"):
        os.environ["SUMO_HOME"] = 'usr/share/sumo'
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

    action = np.array([[0, 13,  0, 12, 12],
                       [25, 0, 25,  0, 25],
                       [ 0, 62, 0, 64, 62],
                       [10,  0, 10, 0, 10],
                       [20, 20, 20, 20, 0]])
    background =np.array([[  0, 931,   0, 764, 695],
                            [670,   0, 460,   0,  21],
                            [  0,  27,   0, 969, 351],
                            [430,   0, 831,   0, 334],
                            [715, 352, 995, 968,   0]])
    setting = {"EDGE_TRAFFIC": background}
    demo = SumoEnv(setting, sim_steps=2000)
    print(demo.simulate(action, is_display=False))