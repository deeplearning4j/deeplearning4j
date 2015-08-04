package org.deeplearning4j.caffe.dag;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.dag.Node;

import java.util.HashMap;
import java.util.Map;

/**
 * @author jeffreytang
 */
@Data
@NoArgsConstructor
@ToString(of = {"name", "type"})
public class CaffeNode implements Node {
    String name;
    NodeType type;
    public enum NodeType { LAYER, BLOB }
    // Meta data about the node
    Map<String, Object> data = new HashMap<>();

    // Constructor
    public CaffeNode(NodeType type, String name) {
        this.type = type;
        this.name = name;
    }
}
