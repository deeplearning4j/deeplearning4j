package org.deeplearning4j.nearestneighbor.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Created by agibsonccc on 12/24/16.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class Base64NDArrayBody implements Serializable {
    private String ndarray;
    private int k;
    private boolean forceFillK;
}
