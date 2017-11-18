package org.nd4j.graph.intermediate;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class is basic scope representation: as ordered list of ops
 * @author raver119@gmail.com
 */
@EqualsAndHashCode
public class TScope {
    private Map<Integer, TOp> numericMap = new HashMap<>();
    private Map<String, TOp> symbolicMap = new HashMap<>();

    @Getter private List<TOp> nodes = new ArrayList<>();
    @Getter private int id;
    @Getter private String name;


    public TScope(int id, @NonNull String name) {
        this.id = id;
        this.name = name;
    }

    /**
     * This method adds operation to the end of the scope
     *
     * @param node
     */
    public void addNode(@NonNull TOp node) {
        nodes.add(node);
        if (node.getId() != 0)
            numericMap.put(node.getId(), node);

        if (node.getName() != null && !node.getName().isEmpty())
            symbolicMap.put(node.getName(), node);
    }


    public TOp getNode(@NonNull String name) {
        return symbolicMap.get(name);
    }


    public TOp getNode(int id) {
        return numericMap.get(id);
    }


    /**
     * This method returns last node of this scope
     * @return
     */
    public TOp lastNode() {
        return nodes.get(nodes.size() - 1);
    }

    /**
     * This method returns number of nodes in this scope
     *
     * @return
     */
    public int size() {
        return nodes.size();
    }
}
