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

package org.nd4j.linalg.api.ops.impl.image;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * CropAndResize Op
 * @author Alex Black
 */
@NoArgsConstructor
public class CropAndResize extends DynamicCustomOp {


    public CropAndResize(@NonNull INDArray image, @NonNull INDArray cropBoxes, @NonNull INDArray boxIndices, @NonNull INDArray cropOutSize, double extrapolationValue) {
        super(new INDArray[]{image, cropBoxes, boxIndices, cropOutSize}, null);
        Preconditions.checkArgument(image.rank() == 4, "Input image must be rank 4 with shape [batch, height, width, channels], got %ndShape", image);
        Preconditions.checkArgument(cropBoxes.rank() == 2 && cropBoxes.size(1) == 4, "Crop boxes must be rank 4 with shape [num_boxes, 5], got %ndShape", cropBoxes);
        Preconditions.checkArgument(boxIndices.rank() == 1 && cropBoxes.size(0) == boxIndices.size(0),
                "Box indices must be rank 1 array with shape [num_boxes] (same as cropBoxes.size(0), got array with shape %ndShape", boxIndices);
        this.method = method;
        this.extrapolationValue = extrapolationValue;
        addArgs();


    }

    public enum Method {BILINEAR, NEAREST};
    protected Method method = Method.BILINEAR;
    protected double extrapolationValue = 0.0;

    public CropAndResize(@NonNull SameDiff sameDiff, @NonNull SDVariable image, @NonNull SDVariable cropBoxes, @NonNull SDVariable boxIndices,
                         @NonNull SDVariable cropOutSize, @NonNull Method method, double extrapolationValue){
        super(sameDiff, new SDVariable[]{image, cropBoxes, boxIndices, cropOutSize});
        this.method = method;
        this.extrapolationValue = extrapolationValue;
        addArgs();
    }

    public CropAndResize(@NonNull INDArray image, @NonNull INDArray cropBoxes, @NonNull INDArray boxIndices,
                         @NonNull INDArray cropOutSize, double extrapolationValue,
                         INDArray output){
        super(new INDArray[]{image, cropBoxes, boxIndices, cropOutSize}, null);
        Preconditions.checkArgument(image.rank() == 4, "Input image must be rank 4 with shape [batch, height, width, channels], got %ndShape", image);
        Preconditions.checkArgument(cropBoxes.rank() == 2 && cropBoxes.size(1) == 4, "Crop boxes must be rank 4 with shape [num_boxes, 5], got %ndShape", cropBoxes);
        Preconditions.checkArgument(boxIndices.rank() == 1 && cropBoxes.size(0) == boxIndices.size(0),
                "Box indices must be rank 1 array with shape [num_boxes] (same as cropBoxes.size(0), got array with shape %ndShape", boxIndices);
        this.method = method;
        this.extrapolationValue = extrapolationValue;
        addArgs();
        outputArguments.add(output);
    }



    

    @Override
    public String opName() {
        return "crop_and_resize";
    }

    @Override
    public String tensorflowName() {
        return "CropAndResize";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        String method = attributesForNode.get("method").getS().toStringUtf8();
        if(method.equalsIgnoreCase("nearest")){
            this.method = Method.NEAREST;
        } else {
            this.method = Method.BILINEAR;
        }

        if(attributesForNode.containsKey("extrapolation_value")){
            extrapolationValue = attributesForNode.get("extrapolation_value").getF();
        }

        addArgs();
    }

    protected void addArgs() {
        addIArgument(method == Method.BILINEAR ? 0 : 1);
        addTArgument(extrapolationValue);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //TODO we can probably skip this sometimes...
        List<SDVariable> out = new ArrayList<>();
        for(SDVariable v : args()){
            out.add(sameDiff.zerosLike(v));
        }
        return out;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 4,
                "Expected 4 input datatypes for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(DataType.FLOAT);   //TF import: always returns float32...
    }
}
