package org.deeplearning4j.nn.graph;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;

@AllArgsConstructor
@Data
@EqualsAndHashCode
public class Edge {

    private final String fromName;
    private final int fromIndex;
    private final int fromOutputNum;

    private final String toName;
    private final int toIndex;
    private final int toInputNum;

}
