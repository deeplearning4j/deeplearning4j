package org.nd4j.linalg.api.ops.random.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;

/**
 * This Op implements numpy.choice method
 * It fills Z from source, following probabilities for each source element
 *
 * @author raver119@gmail.com
 */
public class Choice extends BaseRandomOp {

    public Choice() {
        // no-op
    }

    public Choice(INDArray source, INDArray probabilities, INDArray z) {
        if (source.length() != probabilities.length())
            throw new IllegalStateException("From & probabilities length mismatch: " + source.length() + "/" + probabilities.length());

        init(source, probabilities, z, z.length());
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public int opNum() {
        return 5;
    }

    @Override
    public String name() {
        return "choice";
    }
}
