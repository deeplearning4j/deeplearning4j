package org.deeplearning4j.dag;

import lombok.Data;

import java.util.HashMap;
import java.util.Map;

/**
 * @author jeffreytang
 */
@Data
public class CaffeNode {
    // Constructor of a CaffeNode takes the type and the name
    public CaffeNode(NodeType type, String name) {
        this.type = type;
        this.name = name;
    }
    // Type of either LAYER or BLOB
    NodeType type;
    public enum NodeType { LAYER, BLOB }
    // Name of the Node
    String name;
    // Meta data about the node
    Map<String, Object> data = new HashMap<>();
}
