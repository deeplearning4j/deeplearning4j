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

package org.nd4j.linalg.api.ops.impl.reduce;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.MeanBp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.VarianceBp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.primitives.Ints;
import org.nd4j.shade.guava.primitives.Longs;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

public class Moments extends DynamicCustomOp {

    private boolean keepDims;

    public Moments() {
    }

    public Moments(@NonNull INDArray input, boolean keepDims, long... dimensions) {
        super(new INDArray[]{input}, null);
        this.dimensions = dimensions;
        this.keepDims = keepDims;
        addArgs();
    }

    public Moments(@NonNull INDArray input, long... dimensions) {
        super(new INDArray[]{input}, null);
        this.dimensions = dimensions;
        addArgs();
    }

    public Moments(SameDiff sameDiff, SDVariable input) {
        this(sameDiff, input, null);
        addArgs();
    }

    public Moments(SameDiff sameDiff, SDVariable input, long[] axes) {
        super(null, sameDiff, new SDVariable[] {input}, false);
        this.dimensions = axes;
        addArgs();
    }

    public Moments(INDArray in, INDArray outMean, INDArray outStd, long... axes) {
        super(null, new INDArray[]{in}, new INDArray[]{outMean, outStd}, null, axes);
        this.dimensions = axes;
        addArgs();
    }

    public Moments(INDArray input, long[] axes, boolean keepDims) {
        super(null,new INDArray[]{input},null);
        this.keepDims = keepDims;
        this.dimensions = axes;
        addArgs();
    }

    public Moments(INDArray input, INDArray axes, boolean keepDims) {
        super(null,new INDArray[]{input,axes},null);
        this.keepDims = keepDims;
        addArgs();
    }

    public Moments(SameDiff sd, SDVariable input, long[] axes, boolean keepDims) {
        super(null,sd,new SDVariable[]{input},false);
        this.keepDims = keepDims;
        this.dimensions = axes;
        addArgs();
    }

    public Moments(SameDiff sd, SDVariable input, SDVariable axes, boolean keepDims) {
        super(null,sd,new SDVariable[]{input,axes},false);
        this.keepDims = keepDims;
        addArgs();
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
    }

    @Override
    public String opName() {
        return "moments";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        SDVariable dLdMean = grad.get(0);
        SDVariable dLdVar = grad.get(1);        //Note: non-bias-corrected variance
        if(dimensions != null) {
            SDVariable meanBp = new MeanBp(sameDiff, arg(), dLdMean, keepDims, dimensions).outputVariable();
            SDVariable varBp = new VarianceBp(sameDiff, arg(), dLdVar, false, keepDims, dimensions).outputVariable();
            return Collections.singletonList(meanBp.add(varBp));

        } else if(numIArguments() > 0) {
            long[] newDimensions = Longs.toArray(this.iArguments);
            this.dimensions = newDimensions;
            SDVariable meanBp = new MeanBp(sameDiff, arg(), dLdMean, keepDims, newDimensions).outputVariable();
            SDVariable varBp = new VarianceBp(sameDiff, arg(), dLdVar, false, keepDims,newDimensions).outputVariable();
            return Collections.singletonList(meanBp.add(varBp));

        } else if(numInputArguments() > 1) {
            SDVariable meanBp = new MeanBp(sameDiff, arg(), dLdMean, keepDims, arg(1)).outputVariable();
            SDVariable varBp = new VarianceBp(sameDiff, arg(), dLdVar, false, keepDims, arg(1)).outputVariable();
            return Collections.singletonList(meanBp.add(varBp));
        } else {
            SDVariable meanBp = new MeanBp(sameDiff, arg(), dLdMean, keepDims, dimensions).outputVariable();
            SDVariable varBp = new VarianceBp(sameDiff, arg(), dLdVar, false, keepDims, dimensions).outputVariable();
            return Collections.singletonList(meanBp.add(varBp));
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected 1 datatype for %s, got %s", getClass(), dataTypes);
        if(dataTypes.get(0).isFPType())
            return Arrays.asList(dataTypes.get(0), dataTypes.get(0));
        return Arrays.asList(Nd4j.defaultFloatingPointType(), Nd4j.defaultFloatingPointType());
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> properties = new HashMap<>();
        properties.put("keepDims",keepDims);
        properties.put("dimensions",dimensions);
        return properties;
    }

    protected void addArgs() {
        addBArgument(keepDims);
        if(dimensions != null && dimensions.length > 0) {
            if(dimensions.length != 1 || dimensions[0] != Integer.MAX_VALUE) {
                //Integer.MAX_VALUE means "full array" but here no dimension args == full array
                addIArgument(dimensions);
            }
        }
    }
}
