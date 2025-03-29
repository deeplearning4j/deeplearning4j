/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.common.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

@Slf4j
public abstract class BaseReduceOp extends BaseOp implements ReduceOp {
    @Setter @Getter
    protected boolean keepDims = false;
    @Setter @Getter
    protected boolean isComplex = false;
    @Setter @Getter
    protected boolean isEmptyReduce = false;
    @Setter @Getter
    protected SDVariable dimensionVariable;
    private String dimensionVariableName;

    public BaseReduceOp(SameDiff sameDiff,
                        SDVariable i_v,
                        long[] dimensions, boolean keepDims) {
        super(sameDiff, null);
        if (i_v != null) {
            if(dimensions == null || dimensions.length < 1)
                dimensions = new long[] {Integer.MAX_VALUE};

            this.dimensions = dimensions;
            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v, this);
            this.keepDims = keepDims;
            this.xVertexId = i_v.name();
            sameDiff.addArgsFor(new String[]{xVertexId},this);
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }

        defineDimensions(dimensions);
    }

    public BaseReduceOp(SameDiff sameDiff,
                        SDVariable i_v,
                        SDVariable i_v2,
                        long[] dimensions, boolean keepDims) {
        super(sameDiff,null);
        if (i_v != null) {
            if(dimensions == null || dimensions.length < 1)
                dimensions = new long[] {Integer.MAX_VALUE};

            this.dimensions = dimensions;

            this.xVertexId = i_v.name();
            this.yVertexId = i_v2.name();
            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v, this);
            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v2, this);
            this.keepDims = keepDims;
            sameDiff.addArgsFor(new String[]{xVertexId,yVertexId},this);

        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }

        defineDimensions(dimensions);
    }


    public BaseReduceOp(SameDiff sameDiff,
                        SDVariable i_v) {
        this(sameDiff, i_v, false);
    }


    public BaseReduceOp(SameDiff sameDiff,
                        SDVariable i_v,
                        long[] dimensions) {
        this(sameDiff,i_v,dimensions,false);

    }

    public BaseReduceOp(SameDiff sameDiff,
                        SDVariable i_v,
                        SDVariable i_v2,
                        long[] dimensions) {
        this(sameDiff,i_v,i_v2,dimensions,false);
    }









    //Special constructors for allowing dimensions to be an SDVariable

    public BaseReduceOp(SameDiff sameDiff,
                        SDVariable i_v,
                        boolean keepDims) {
        super(sameDiff, null);
        if (i_v != null) {
            if(dimensions == null || dimensions.length < 1)
                dimensions = new long[] {Integer.MAX_VALUE};

            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v, this);
            this.keepDims = keepDims;
            this.xVertexId = i_v.name();
            sameDiff.addArgsFor(new String[]{xVertexId},this);
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }

        defineDimensions(dimensions);
    }

    public BaseReduceOp(SameDiff sameDiff,
                        SDVariable i_v,
                        SDVariable dimensions,
                        boolean keepDims) {
        super(sameDiff,null);

        this.dimensionVariable = dimensions;


        this.xVertexId = i_v.name();
        this.yVertexId = dimensions.name();
        SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v, this);
        SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, dimensions, this);
        this.keepDims = keepDims;
        sameDiff.addArgsFor(new String[]{xVertexId,yVertexId},this);

    }


    public BaseReduceOp(SameDiff sameDiff,
                        SDVariable i_v,
                        SDVariable i_v2) {
        this(sameDiff,i_v,i_v2,false);
    }






    public BaseReduceOp() {}


    public BaseReduceOp(INDArray x, INDArray y, INDArray z, boolean keepDims, long[] dimensions) {
        super(x, y, z);
        this.keepDims = keepDims;
        this.dimensions = dimensions;
        defineDimensions(dimensions);
    }

    public BaseReduceOp(INDArray x, long... dimensions) {
        this(x, null, dimensions);
    }

    public BaseReduceOp(INDArray x, boolean keepDims, long... dimensions) {
        this(x, null, dimensions);
        this.keepDims = keepDims;
    }

    public BaseReduceOp(INDArray x, INDArray y, long... dimensions) {
        this(x, y, null, dimensions);
    }

    public BaseReduceOp(INDArray x, INDArray y, INDArray z, long... dimensions) {
        this(x, y, z, false, dimensions);
    }

    public BaseReduceOp(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    public BaseReduceOp(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions) {
        this(sameDiff,i_v,dimensions,false);
    }

    @Override
    public INDArray noOp() {
        if (z != null && x != z)
            return z().assign(x.reshape(z.shape()));
        else {
            //Need to take into account shapes: for example, [1,3].sum(0) -> [3]
            //Or [1,1,1,1].sum(0,2,3) -> [1]
            if(keepDims){
                return x().dup(x().ordering());
            } else {
                long[] shape = x.shape();
                if(dimensions == null || Shape.isWholeArray(shape, dimensions)){
                    //Return scalar
                    return x.reshape().dup();
                } else {
                    //Strip out size 1 dimensions
                    long[] outShape = ArrayUtil.removeIndex(shape, dimensions);
                    return x.dup('c').reshape('c', outShape);
                }
            }
        }
    }

    @Override
    public boolean isKeepDims() {
        return keepDims;
    }


    public abstract List<DataBuffer> calculateOutputShape();


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if (!attributesForNode.containsKey("axis") && !hasReductionIndices(nodeDef)) {
            this.dimensions = new long[] { Integer.MAX_VALUE };
        }   //Otherwise: dimensions are dynamically set during execution in InferenceSession

        if(attributesForNode.containsKey("keep_dims")) {
            val keepDims = attributesForNode.get("keep_dims").getB();
            this.keepDims = keepDims;
        }
        defineDimensions(this.dimensions);
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
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {

    }

    @Override
    public boolean isComplexAccumulation() {
        return isComplex;
    }

    @Override
    public void setDimensions(long... dimensions) {
        this.dimensions = dimensions;
        defineDimensions(dimensions);
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(properties.containsKey("isEmptyReduce")) {
            Boolean isEmptyReduce = getBooleanFromProperty("isEmptyReduce",properties);
            this.isEmptyReduce = isEmptyReduce;
        }

        if(properties.containsKey("keepDims")) {
            Boolean keepDims = getBooleanFromProperty("keepDims",properties);
            this.keepDims = keepDims;

        }


        if(properties.containsKey("isComplex")) {
            Boolean isComplex = getBooleanFromProperty("isComplex",properties);
            this.isComplex = isComplex;
        }

        if(properties.containsKey("dimensionz")) {
            INDArray array = (INDArray) properties.get("dimensionz");
            this.dimensionz = array;
        }

        if(properties.containsKey("dimensionVariable") && properties.get("dimensionVariable") != null) {
            String varName = properties.get("dimensionVariable").toString();
            this.dimensionVariableName = varName;
        }
    }

    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
        if(dimensionVariableName != null)
            this.dimensionVariable = sameDiff.getVariable(dimensionVariableName);

    }
}
