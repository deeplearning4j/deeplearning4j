package org.nd4j.imports.converters;

import lombok.NonNull;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.nd4j.linalg.api.ops.Op;

/**
 * This interface describes
 *
 * @author raver119@gmail.com
 */
public interface ExternalNode<T> {
    /**
     * This method returns given TF node as TNode
     *
     * @return
     */
    TNode asIntermediateRepresentation(@NonNull T node, @NonNull TGraph graph);

    /**
     * This method returns given TF node as ND4j Op
     *
     * @return
     */
    Op  asExecutableOperation(@NonNull T node, @NonNull TGraph graph);


    String opName();
}
