package org.deeplearning4j.ui.flow.beans;

import lombok.Data;

import java.io.Serializable;

/**
 * @author raver119@gmail.com
 */
@Data
public class Coords implements Serializable {
    private int x;
    private int y;

    public Coords() {

    }

    public Coords(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public static Coords makeCoors(int x, int y) {
        return new Coords(x, y);
    }
}
