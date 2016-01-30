package org.deeplearning4j.ui.flow.beans;

import lombok.Data;

import java.io.Serializable;

/**
 * This bean describes abstract layer and it's connections
 *
 * @author raver119@gmail.com
 */
@Data
public class LayerInfo implements Serializable {
    private long id;

    private String name;

    private int x = 0;
    private int y = 0;

    private Description description;

    private LayerInfo connectedTo;
}
