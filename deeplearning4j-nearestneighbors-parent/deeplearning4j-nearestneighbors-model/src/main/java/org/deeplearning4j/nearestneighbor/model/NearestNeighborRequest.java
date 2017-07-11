package org.deeplearning4j.nearestneighbor.model;

import lombok.Data;

import java.io.Serializable;

/**
 * Created by agibsonccc on 4/26/17.
 */
@Data
public class NearestNeighborRequest implements Serializable {
    private int k;
    private int inputIndex;

}
