package org.deeplearning4j.nearestneighbor.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.List;

/**
 * Created by agibsonccc on 4/27/17.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NearestNeighborsResults implements Serializable {
    private List<NearestNeighborsResult> results;

}
