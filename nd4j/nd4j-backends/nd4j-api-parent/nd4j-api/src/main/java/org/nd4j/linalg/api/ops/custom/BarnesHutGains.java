package org.nd4j.linalg.api.ops.custom;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

/**
 * This op calculates gains - data used internally by Barnes-Hut-TSNE algorithm.
 *
 * @author alexander.stoyakin@gmail.com
 */
public class BarnesHutGains extends DynamicCustomOp {

    public BarnesHutGains(){ }

    public BarnesHutGains(INDArray output, INDArray input, INDArray gradx, INDArray epsilon) {

        inputArguments.add(input);
        inputArguments.add(gradx);
        inputArguments.add(epsilon);

        outputArguments.add(output);
    }

    @Override
    public String opName() {
        return "barnes_gains";
    }
}
