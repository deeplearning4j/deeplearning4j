/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ShapeOp;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Reshape function
 *
 * @author Adam Gibson
 */
@Slf4j
public class Reshape extends ShapeOp {

    private int[] shape;

    public Reshape(SameDiff sameDiff, DifferentialFunction i_v,int[] shape) {
        super(sameDiff, i_v, false);
        this.shape = shape;
    }

    public Reshape(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, Object[] extraArgs, int[] shape1) {
        super(sameDiff, i_v, shape, false, extraArgs);
        this.shape = shape1;
    }

    public Reshape() {}

    public Reshape(INDArray x, INDArray z) {
        super(x, z);
    }

    public Reshape(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Reshape(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Reshape(INDArray x) {
        super(x);
    }

    @Override
    public void exec(int... dimensions) {
        exec();
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public void exec() {
        if(x != z) {
            z.assign(x.reshape(shape));
        }
        else {
            this.z = x.reshape(shape);
        }

    }


    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "reshape";
    }

    @Override
    public String onnxName() {
        return "Reshape";
    }

    @Override
    public String tensorflowName() {
        return "reshape";
    }


    @Override
    public TOp asIntermediateRepresentation(NodeDef node, TGraph graph) {
        val tNode = buildBasicNode(node, graph);

        // in reshape operation we replace second input, and replace it with extra args
        log.debug("TOp inputs: {}", tNode.getInputs());
        val shapeIndex = tNode.getInputs().remove(1);
        val variable = graph.getVariableSpace().getVariable(shapeIndex);

        assert variable != null;
        assert variable.getShape() != null;

        // we know that TF is always C order
        int[] args = ArrayUtils.add(variable.getShape(),  0, (int)'c');


        log.debug("Reshape node_{}, new shape: {}", tNode.getId(), Arrays.toString(args));

        // new shape goes here
        tNode.getOpState().setExtraBits(args);

        return tNode;
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = this;

        return Collections.singletonList(ret);
    }

}
