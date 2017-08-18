package org.deeplearning4j.nearestneighbor.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
/**
 * Created by agibsonccc on 4/26/17.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class NearestNeighborsResult {
    public NearestNeighborsResult(int index, double distance) {
        this(index, distance, null);
    }

    private int index;
    private double distance;
    private String label;
}
