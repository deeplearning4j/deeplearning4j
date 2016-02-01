package org.deeplearning4j.ui.flow.beans;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;

/**
 * This bean describes abstract layer and it's connections
 *
 * @author raver119@gmail.com
 */
@Data
public class LayerInfo implements Serializable {
    private final static long serialVersionUID = 119L;
    private long id;
    private String name;
    private String layerType;

     //   grid coordinates. row & column
    private int x = 0;
    private int y = 0;

    private Description description;

    // set of connections as grid coordinates
    private Set<Coords> connections = new HashSet<>();

    public void addConnection(LayerInfo layerInfo) {
        connections.add(Coords.makeCoors(layerInfo.getX(), layerInfo.getY()));
    }

    public void addConnection(int x, int y) {
        connections.add(Coords.makeCoors(x, y));
    }

    public void dropConnection(int x, int y) {
        connections.remove(Coords.makeCoors(x, y));
    }

    public void dropConnections() {
        connections.clear();
    }
}
