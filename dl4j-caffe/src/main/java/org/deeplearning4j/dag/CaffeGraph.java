package org.deeplearning4j.dag;

import lombok.NoArgsConstructor;

import java.util.*;

/**
 * @author jeffreytang
 */
@NoArgsConstructor
public class CaffeGraph {

    // Key of adjacency list points to nodes in the list (from bottom to top)
    Map<CaffeNode, List<CaffeNode>> adjancencyListMap = new HashMap<>();
    Map<String, CaffeNode> name2NodeMap = new HashMap<>();
    Set<CaffeNode> rootNodeSet = new HashSet<>();

    public int graphSize() {
        return name2NodeMap.size();
    }

    public void addEdge(String nameOfNodeA, String nameOfNodeB) {
        CaffeNode nodeA = name2NodeMap.get(nameOfNodeA);
        CaffeNode nodeB = name2NodeMap.get(nameOfNodeB);
        addEdge(nodeA, nodeB);
    }

    public void addEdge(CaffeNode nodeA, CaffeNode nodeB) {
        addNode(nodeA);
        addNode(nodeB);
        adjancencyListMap.get(nodeA).add(nodeB);
    }

    public void addNode(CaffeNode node) {
        if (!name2NodeMap.containsKey(node.getName())) {
            name2NodeMap.put(node.getName(), node);
        }
    }

    public void addRootNode(CaffeNode rootNode) {
        rootNodeSet.add(rootNode);
    }

    public CaffeNode getNode(String nameOfNode) {
        return name2NodeMap.get(nameOfNode);
    }

    public List<CaffeNode> getNodeAdjacencyList(String nameOfNode) {
        return adjancencyListMap.get(getNode(nameOfNode));
    }
}
