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
    for tripinfo in trapinfos:
           begin_node=int(tripinfo.getAttribute('id')[1])
           end_node=int(tripinfo.getAttribute('id')[2])
           total_time[begin_node][end_node] += float(tripinfo.getAttribute('duration'))
    for i in range(row):
        for j in range(column):
            if action[i][j] != 0:
                avg_time[i][j] = total_time[i][j]/action[i][j]
    return avg_time


class SumoEnv:
    """Environment of SUMO used to simulate"""
    def __init__(self, setting):
        self.edge_traffic = setting['EDGE_TRAFFIC']
        self.path = os.getcwd()
        if self.path[-1] == 'n':    # If current path is .../gnn, then add the directory path
            self.path += '/transition_model/'

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
        vType = dom.createElement('vType')
        vType.setAttribute("id","CarA")
        vType.setAttribute("length","5")
        vType.setAttribute("maxSpeed","22")
        routes.appendChild(vType)

        row, column = self.action.shape
        for i in range(row):
            for j in range(column):
                if action[i][j]!=0:
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
        for step in range(0, 2000):
            traci.simulationStep()
        traci.close()
        output = Duration(self.path+'info/info.xml', self.action)
        return output


if __name__ == "__main__":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    action =np.array([[0, 1, 0, 5, 3],
              [2, 0, 8, 0, 3],
              [0, 5, 0, 2, 1],
              [2, 0, 5, 0, 1],
              [1, 2, 1, 2, 0]])
    demo = SumoEnv()
    print(demo.simulate(action))
