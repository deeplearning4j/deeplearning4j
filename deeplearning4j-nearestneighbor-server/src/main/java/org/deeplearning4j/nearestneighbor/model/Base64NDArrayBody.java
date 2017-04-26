package org.deeplearning4j.nearestneighbor.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Created by agibsonccc on 12/24/16.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class Base64NDArrayBody {
    private String ndarray;
}
