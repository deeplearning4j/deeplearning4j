package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
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

    public Choice(@NonNull INDArray source, @NonNull INDArray probabilities, @NonNull INDArray z) {
        if (source.lengthLong() != probabilities.lengthLong())
            throw new IllegalStateException("From & probabilities length mismatch: " + source.lengthLong() + "/"
                            + probabilities.lengthLong());

        if (probabilities.elementWiseStride() < 1 || source.elementWiseStride() < 1)
            throw new IllegalStateException("Source and probabilities should have element-wise stride >= 1");

        init(source, probabilities, z, z.lengthLong());
        this.extraArgs = new Object[] {0.0};
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
