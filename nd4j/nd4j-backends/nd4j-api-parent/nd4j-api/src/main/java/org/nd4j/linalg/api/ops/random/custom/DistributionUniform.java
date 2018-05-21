package org.nd4j.linalg.api.ops.random.custom;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Map;

/**
 * Uniform distribution wrapper
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DistributionUniform extends DynamicCustomOp {
    private double min = 0.0;
    private double max = 1.0;

    public DistributionUniform() {
        //
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
        addArgs();
    }

    protected void addArgs() {
        addTArgument(min, max);
    }

    @Override
    public String opName() {
        return "randomuniform";
    }

    @Override
    public String tensorflowName() {
        return "RandomUniform";
    }
}
