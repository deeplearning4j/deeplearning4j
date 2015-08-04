package org.deeplearning4j.dag;

import lombok.NoArgsConstructor;

import java.util.*;

/**
 * @author jeffreytang
 */
@NoArgsConstructor
public class Graph {
    // Key of adjacency list points to nodes in the list (from bottom to top)
    private Map<Node, List<Node>> adjacencyListMap = new HashMap<>();
    private Map<String, Node> name2NodeMap = new HashMap<>();
    private Set<Node> rootNodeSet = new HashSet<>();

    public int graphSize() {
        return name2NodeMap.size();
    }

    public void addEdge(String nameOfNodeA, String nameOfNodeB) {
        Node nodeA =  name2NodeMap.get(nameOfNodeA);
        Node nodeB = name2NodeMap.get(nameOfNodeB);
        addEdge(nodeA, nodeB);
    }

    public void addEdge(Node nodeA, Node nodeB) {
        addNode(nodeA);
        addNode(nodeB);
        if (adjacencyListMap.containsKey(nodeA)) {
            adjacencyListMap.get(nodeA).add(nodeB);
        } else {
            List<Node> lst = new ArrayList<>();
            lst.add(nodeB);
            adjacencyListMap.put(nodeA, lst);
        }
    }

    public void addNode(Node node) {
        if (!name2NodeMap.containsKey(node.getName())) {
            name2NodeMap.put(node.getName(), node);
        }
    }

    public void addRootNode(Node rootNode) {
        rootNodeSet.add(rootNode);
    }

    public Node getNode(String nameOfNode) {
        return name2NodeMap.get(nameOfNode);
    }

    public List<Node> getNodeAdjacencyList(String nameOfNode) {
        return adjacencyListMap.get(getNode(nameOfNode));
    }

    @Override
    public String toString() {
        String sizeString = String.format("\tSize: %s\n", graphSize());
        String adjacencyString = "";
        for (Map.Entry<Node, List<Node>> entry : adjacencyListMap.entrySet()) {
            String curNode = entry.getKey().toString();
            String listNodes = Arrays.deepToString(entry.getValue().toArray());
            adjacencyString += String.format("\t%s -> %s\n", curNode, listNodes);
        }
        String[] classList = this.getClass().toString().split("\\.");
        String className = classList[classList.length - 1];
        return String.format("%s {\n%s%s}", className, sizeString, adjacencyString);
    }
}
