package org.nd4j.autodiff.functions;

import org.nd4j.linalg.api.ops.impl.transforms.Variable;

public interface PreEvaluator {
    void update(Variable v);

}
