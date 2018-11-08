/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops;

import com.google.common.primitives.Ints;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Base class for accumulation, initiates the initial entry
 * with respect to the child class. Also contains baseline fields
 * for the over all field with accumulation.
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseAccumulation extends BaseOp implements Accumulation {
    protected Number finalResult;
    @Setter @Getter
    protected boolean keepDims = false;

    // flag for tf imported ops, shows that there's probably one more value appended in axis
    @Setter @Getter
    protected boolean newFormat = false;
    protected boolean isComplex = false;


    public BaseAccumulation(SameDiff sameDiff,
                            SDVariable i_v,
                            int[] dimensions,boolean keepDims) {
        super(sameDiff,new Object[]{dimensions});
        if (i_v != null) {
            if(dimensions == null || dimensions.length < 1)
                dimensions = new int[] {Integer.MAX_VALUE};

            this.dimensions = dimensions;
            f().validateDifferentialFunctionsameDiff(i_v);
            this.keepDims = keepDims;
            this.xVertexId = i_v.getVarName();
            sameDiff.addArgsFor(new String[]{xVertexId},this);
            if(Shape.isPlaceholderShape(i_v.getShape())) {
                sameDiff.addPropertyToResolve(this,i_v.getVarName());
            }

        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }

        this.newFormat = true;
    }

    public BaseAccumulation(SameDiff sameDiff,
                            SDVariable i_v,
                            SDVariable i_v2,
                            int[] dimensions,boolean keepDims) {
        super(sameDiff,new Object[]{dimensions});
        if (i_v != null) {
            if(dimensions == null || dimensions.length < 1)
                dimensions = new int[] {Integer.MAX_VALUE};

            this.dimensions = dimensions;

            this.xVertexId = i_v.getVarName();
            this.yVertexId = i_v2.getVarName();
            f().validateDifferentialFunctionsameDiff(i_v);
            f().validateDifferentialFunctionsameDiff(i_v2);
            this.keepDims = keepDims;
            sameDiff.addArgsFor(new String[]{xVertexId,yVertexId},this);

        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }

        this.newFormat = true;
    }


    public BaseAccumulation(SameDiff sameDiff,
                            SDVariable i_v) {
        this(sameDiff, i_v, null, false);
    }


    public BaseAccumulation(SameDiff sameDiff,
                            SDVariable i_v,
                            int[] dimensions) {
        this(sameDiff,i_v,dimensions,false);

    }

    public BaseAccumulation(SameDiff sameDiff,
                            SDVariable i_v,
                            SDVariable i_v2,
                            int[] dimensions) {
        this(sameDiff,i_v,i_v2,dimensions,false);
    }



    public BaseAccumulation() {}

    /**
     * Initialize with the given
     * input, pairwise transform, result, and number
     * of elements
     *
     * @param x the input
     * @param y the pairwise transform
     * @param z the result
     * @param n the number of elements
     */
    public BaseAccumulation(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        init();
    }

    public BaseAccumulation(INDArray x, INDArray y, INDArray z, boolean newFormat, boolean keepDims, int[] dimensions) {
        super(x, y, z, x.lengthLong());
        this.newFormat = newFormat;
        this.keepDims = keepDims;
        this.dimensions = dimensions;
        init();
    }

    public BaseAccumulation(INDArray x, INDArray y, long n) {
        this(x, y, x, n);
    }

    public BaseAccumulation(INDArray x) {
        this(x, null, x, x.lengthLong());
    }

    public BaseAccumulation(INDArray x, INDArray y) {
        this(x, y, x, x.lengthLong());
        //if (y != null)
        //    LinAlgExceptions.assertSameLength(x, y);
    }

    public BaseAccumulation(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }




    private void init() {
        if (z == null || x == z)
            init(x, y, x, x.lengthLong());
        else
            init(x, y, z, x.lengthLong());
    }

    @Override
    public INDArray noOp() {
        if (z != null && x != z)
            return z().assign(x);
        else
            return x().dup(x().ordering());
    }

    @Override
    public boolean isKeepDims() {
        return keepDims;
    }

    @Override
    public List<long[]> calculateOutputShape() {
        if(args().length < 1) {
            throw new ND4JIllegalStateException("Unable to compute input shape. No arguments found.");
        }

        long[] argShape = arg().getShape();
        if (argShape == null && x() == null) {
            return Collections.emptyList();
        }
        long[] inputShape = (argShape == null ? x().shape() : argShape);

        List<long[]> ret = new ArrayList<>(1);
        val reducedShape = Shape.getReducedShape(inputShape,dimensions, isKeepDims(), newFormat);
        ret.add(reducedShape);
        return ret;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        newFormat = true;

        if (!attributesForNode.containsKey("axis") && !hasReductionIndices(nodeDef)) {
            this.dimensions = new int[] { Integer.MAX_VALUE };
        }
        else if(hasReductionIndices(nodeDef)) {
            NodeDef reductionNode = null;
            for(int i = 0; i < graph.getNodeCount(); i++) {
                if (graph.getNode(i).getName().equals(nodeDef.getName() + "/reduction_indices")) {
                    reductionNode = graph.getNode(i);
                    val arr = TFGraphMapper.getInstance().getNDArrayFromTensor("value", reductionNode, graph);

                    boolean keepAxis = nodeDef.getAttrOrThrow("keep_dims").getB();

                    // keepAxis = false by default
                    //int[] dimensions = ArrayUtils.add(arr.data().asInt(), 0, keepAxis ? 1 : 0);
                    int[] dimensions = arr.data().asInt();

                    this.dimensions = dimensions;
                    break;
                }
            }

            if(reductionNode == null)
                throw new ND4JIllegalStateException("No node found!");
        }
        else {
            val dims = TFGraphMapper.getInstance().getNDArrayFromTensor("axis",nodeDef,graph).data().asInt();
            this.dimensions = dims;
        }

        if(attributesForNode.containsKey("keep_dims")) {
            val keepDims = attributesForNode.get("keep_dims").getB();
            this.keepDims = keepDims;
        }
    }

    protected boolean hasReductionIndices(NodeDef nodeDef) {
        for(int i = 0; i < nodeDef.getInputCount(); i++) {
            if(nodeDef.getInput(i).contains("reduction_indices")) {
                return true;
            }
        }

        return false;
    }


    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        if (!attributesForNode.containsKey("axes")) {
            this.dimensions = new int[] { Integer.MAX_VALUE };
        }
        else {
            val map = OnnxGraphMapper.getInstance().getAttrMap(node);
            val dims = Ints.toArray(map.get("axes").getIntsList());
            this.dimensions = dims;
        }
    }

    @Override
    public void setFinalResult(double value) {
        this.finalResult = value;
    }

    @Override
    public Number getFinalResult() {
        return finalResult;
    }

    @Override
    public double zeroDouble() {
        return 0;
    }

    @Override
    public float zeroFloat() {
        return 0;
    }

    @Override
    public float zeroHalf() {
        return 0;
    }

    @Override
    public Type opType() {
        return Type.REDUCE;
    }

    @Override
    public boolean isComplexAccumulation() {
        return isComplex;
    }
}
