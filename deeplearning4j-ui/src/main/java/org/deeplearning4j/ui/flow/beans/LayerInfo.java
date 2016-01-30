package org.deeplearning4j.ui.flow.beans;

import lombok.Data;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;

import java.io.Serializable;
import java.util.Set;

/**
 * This bean describes abstract layer and it's connections
 *
 * @author raver119@gmail.com
 */
@Data
public class LayerInfo implements Serializable {
    private long id;

    private String name;

    private Layer.Type layerType;

    /*
        grid coordinates. row & column
     */
    private int x = 0;
    private int y = 0;

    private Description description;

    // set of connections as grid coordinates
    private Set<Pair<Integer, Integer>> connections;
}
