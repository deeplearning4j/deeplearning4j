package org.nd4j.imports.converters;

import lombok.NonNull;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;

public interface NodeMapper<T> {

    TNode asIntermediate(@NonNull T node, @NonNull TGraph graph);
}
