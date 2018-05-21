package org.deeplearning4j.nn.graph.util;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

import java.util.Map;

/**
 * Simple helper class for ComputationGraph topological sort and vertex index/name + name/index mapping
 *
 * @author Alex Black
 */
@Data
@AllArgsConstructor
@Builder
public class GraphIndices {
    private int[] topologicalSortOrder;
    private Map<String,Integer> nameToIdx;
    private Map<Integer,String> idxToName;
}
