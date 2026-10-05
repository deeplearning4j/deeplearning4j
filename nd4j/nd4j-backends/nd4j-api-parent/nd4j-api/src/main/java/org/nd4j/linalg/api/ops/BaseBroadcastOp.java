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

import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@NoArgsConstructor
@Slf4j
public abstract class BaseBroadcastOp extends BaseOp implements BroadcastOp {

    protected long[] dimension;


    public BaseBroadcastOp(SameDiff sameDiff,
                           SDVariable i_v1,
                           SDVariable i_v2,
                           long[] dimension) {
        this(sameDiff, i_v1, i_v2, false, dimension);
    }

    public BaseBroadcastOp(SameDiff sameDiff,
                           SDVariable i_v1,
                           SDVariable i_v2,
                           boolean inPlace,
                           long[] dimension) {
        super(sameDiff, inPlace, new Object[]{i_v2});
        if (i_v1 != null && i_v2 != null) {
            this.sameDiff = sameDiff;
            this.inPlace = inPlace;
            this.dimension = dimension;
            sameDiff.addArgsFor(new SDVariable[]{i_v1,i_v2},this);
        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }
    }

    public BaseBroadcastOp(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    public BaseBroadcastOp(SameDiff sameDiff,
                           SDVariable i_v1,
                           SDVariable i_v2,
                           long[] dimension,
                           Object[] extraArgs) {
        super(sameDiff, extraArgs);
        this.dimension = dimension;
        if (i_v1 != null && i_v2 != null) {
            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v1, this);
            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v2, this);

            this.sameDiff = sameDiff;
            sameDiff.addArgsFor(new SDVariable[]{i_v1,i_v2},this);

        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }


    }


    public BaseBroadcastOp(SameDiff sameDiff, SDVariable i_v, long[] dimension, boolean inPlace) {
        this(sameDiff, i_v, i_v.getShape(), inPlace, dimension, null);
    }

    public BaseBroadcastOp(SameDiff sameDiff,
                           SDVariable i_v,
                           long[] shape,
                           boolean inPlace,
                           long[] dimension,
                           Object[] extraArgs) {
        super(sameDiff, inPlace, extraArgs);
        this.dimension = dimension;
        if (i_v != null) {
            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v, this);
            sameDiff.addArgsFor(new SDVariable[]{i_v},this);


        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }


    }


    public BaseBroadcastOp(SameDiff sameDiff,
                           SDVariable i_v,
                           long[] dimension,
                           Object[] extraArgs) {
        this(sameDiff, i_v, i_v.getShape(), false, dimension, extraArgs);
    }

    public BaseBroadcastOp(INDArray x, INDArray y, INDArray z, long... dimension) {
        super(x, y, z);
        Broadcast.validateBroadcastDims(x, y, z, dimension);
        setDimension(dimension);
    }

    @Override
    public Type opType() {
        return Type.BROADCAST;
    }

    /**
     * Calculate the output shape for this op
     *
     * @return
     */
    public List<DataBuffer> calculateOutputShape() {
        if(x == null || y == null)
            return Collections.emptyList();

        long[] shapeX = x.shape();
        long[] shapeY = y.shape();

        return Collections.singletonList(Nd4j.createBuffer(LongShapeDescriptor
                .fromShape(Shape.broadcastOutputShape(shapeX, shapeY),
                Shape.pickPairwiseDataType(x.dataType(), y.dataType()))
                .toShapeInfo()));
    }


    @Override
    public long[] getDimension() {
        if (dimension == null) {
            if (x != null && y != null) {
                dimension = Shape.getBroadcastDimensions(x.shape(), y.shape());
            } else if (sameDiff != null) {
                dimension = Shape.getBroadcastDimensions(larg().getShape(), rarg().getShape());
            }
        }
        return dimension;
    }


    @Override
    public long[] getDimension(OpContext context) {
        if (context == null || dimension != null) {
            return getDimension();
        }
        INDArray input = context.getInputArray(0);
        INDArray broadcast = context.getInputArray(1);
        if (input == null || broadcast == null) {
            throw new IllegalArgumentException("Broadcast execution requires X and Y arrays");
        }
        return Shape.getBroadcastDimensions(input.shape(), broadcast.shape());
    }

    @Override
    public void setDimension(long... dimension) {
        if (x != null) {
            // Reject invalid updates before replacing the op's existing definition.
            Broadcast.normalizeBroadcastDimensions(x.rank(), dimension);
        }
        this.dimension = dimension == null ? null : dimension.clone();
        // A symbolic/context-only op has no input rank yet. Materialize at validation instead.
        if (x != null) {
            updateDimensions(x.rank(), getDimension());
        }
    }

    private void updateDimensions(int rank, long[] axes) {
        long[] normalized = Broadcast.normalizeBroadcastDimensions(rank, axes);
        if (!Arrays.equals(dimensions, normalized)) {
            dimensions = normalized;
            dimensionz = Shape.ndArrayDimFromLong(normalized);
        }
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
    }



    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {

    }

    @Override
    public boolean validateDataTypes(boolean experimentalMode) {
        return validateDataTypes(null, experimentalMode);
    }

    @Override
    public boolean validateDataTypes(OpContext context, boolean experimentalMode) {
        BroadcastOp.super.validateDataTypes(context, experimentalMode);
        INDArray input = context == null ? x() : context.getInputArray(0);
        updateDimensions(input.rank(), getDimension(context));
        return true;
    }

    @Override
    public Type getOpType() {
        return Type.BROADCAST;
    }
}
