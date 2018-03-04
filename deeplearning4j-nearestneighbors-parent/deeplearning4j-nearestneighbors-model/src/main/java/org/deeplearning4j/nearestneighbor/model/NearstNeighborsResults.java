package org.deeplearning4j.nearestneighbor.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.List;

/**
 * @deprecated Feb 2018 due to typo - use {@link NearestNeighborsResults}
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Deprecated
public class NearstNeighborsResults implements Serializable {
    private List<NearestNeighborsResult> results;
}
