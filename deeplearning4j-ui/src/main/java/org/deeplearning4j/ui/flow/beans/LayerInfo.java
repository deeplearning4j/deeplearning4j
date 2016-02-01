package org.deeplearning4j.ui.flow.beans;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
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

    private String color;

     //   grid coordinates. row & column
    private int x = 0;
    private int y = 0;

    private Description description;

    // set of connections as grid coordinates
    private List<Coords> connections = new ArrayList<>();

    public void addConnection(LayerInfo layerInfo) {
        if (!connections.contains(Coords.makeCoors(layerInfo.getX(), layerInfo.getY()))) {
            connections.add(Coords.makeCoors(layerInfo.getX(), layerInfo.getY()));
        }
    }

    public void addConnection(int x, int y) {
        if (!connections.contains(Coords.makeCoors(x, y)))
            connections.add(Coords.makeCoors(x, y));
    }

    public void dropConnection(int x, int y) {
        connections.remove(Coords.makeCoors(x, y));
    }

    public void dropConnections() {
        connections.clear();
    }
}
