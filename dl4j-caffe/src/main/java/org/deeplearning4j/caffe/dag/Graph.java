package org.deeplearning4j.caffe.dag;

/**
 * @author jeffreytang
 */

import io.netty.util.internal.ConcurrentSet;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.*;

@NoArgsConstructor
@Data
public class Graph<T extends Node> {
    // Key of adjacency list points to nodes in the list (from bottom to top)
    private Map<T, Set<T>> adjacencyListMap = new HashMap<>();
    private Set<T> startNodeSet = new ConcurrentSet<>();
    private Set<T> endNodeSet = new ConcurrentSet<>();

    public int graphSize() {
        return adjacencyListMap.size();
    }

    public void addNode(T node) {
        if (!adjacencyListMap.containsKey(node)) {
            adjacencyListMap.put(node, new HashSet<T>());
        }
    }

    public void addEdge(T nodeA, T nodeB) {
        addNode(nodeA);
        addNode(nodeB);
        adjacencyListMap.get(nodeA).add(nodeB);
    }

    public void addEdges(T nodeA, List<T> nodeList) {
        for (T nodeB : nodeList) {
            addEdge(nodeA, nodeB);
        }
    }

    /**
     * Return if there is an edge of nodeA pointing towards nodeB
     *
     * @param nodeA The node pointing to another node
     * @param nodeB The node being pointed to
     * @return Boolean
     */
    public Boolean edgeBetween(T nodeA, T nodeB) {
        Boolean ret = false;
        if (adjacencyListMap.containsKey(nodeA)) {
            for (Node node : adjacencyListMap.get(nodeA)) {
                if (node.equals(nodeB))
                    ret = true;
            }
        }
        return ret;
    }

    public Set<T> getNeighbors(T node) {
        return adjacencyListMap.get(node);
    }

    /**
     * Return if the node is in the graph
     *
     * @param node The query node
     * @return Boolean
     */
    public Boolean hasNode(T node) {
        Boolean ret = false;
        if (adjacencyListMap.containsKey(node)) {
            ret = true;
        }
        return ret;
    }

    public void addStartNode(T node) { startNodeSet.add(node); }

    public void addEndNode(T node) { endNodeSet.add(node); }

    public void removeStartNode(T node) { startNodeSet.remove(node); }

    public void removeEndNode(T node) { endNodeSet.remove(node); }

    /**
     * Return a list of nodes with the given name
     *
     * @param nameOfNode The query name
     * @return List of Node Objects
     */
    public List<Node> getNodesWithName(String nameOfNode) {
        List<Node> matchedNodeList = new ArrayList<>();
        for(Node node : adjacencyListMap.keySet()) {
            if (node.getName().equals(nameOfNode))
                matchedNodeList.add(node);
        }
        return matchedNodeList;
    }

    /**
     * Returns a list of nodes a particular node points to
     *
     * @param node The query node
     * @return List of neighbor Node Objects
     */
    public Set<T> getNextNodes(T node) {
        return adjacencyListMap.get(node);
    }

    /**
     * Turn Graph Object to customized String
     *
     * @return String representation of graph
     */
    @Override
    public String toString() {
        String sizeString = String.format("\tSize: %s\n", graphSize());
        String adjacencyString = "";
        for (Map.Entry<T, Set<T>> entry : adjacencyListMap.entrySet()) {
            String curNode = entry.getKey().toString();
            String listNodes = Arrays.deepToString(entry.getValue().toArray());
            adjacencyString += String.format("\t%s -> %s\n", curNode, listNodes);
        }
        String[] classList = this.getClass().toString().split("\\.");
        String className = classList[classList.length - 1];
        return String.format("%s {\n%s%s}", className, sizeString, adjacencyString);
    }
}