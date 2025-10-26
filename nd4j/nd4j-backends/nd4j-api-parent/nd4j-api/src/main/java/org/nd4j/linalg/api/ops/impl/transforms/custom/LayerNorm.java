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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.shade.guava.primitives.Ints;
import org.nd4j.shade.guava.primitives.Longs;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


@NoArgsConstructor
public class LayerNorm extends DynamicCustomOp {

    private boolean noBias = false;
    private boolean channelsFirst;

    public LayerNorm(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable gain, SDVariable bias, boolean channelsFirst, long... dimensions) {
        super(null, sameDiff, wrapFilterNull(input, gain, bias), false);
        this.noBias = bias == null;
        this.channelsFirst = channelsFirst;
        setDimensions(dimensions);
    }

    public LayerNorm(SameDiff sameDiff, SDVariable input, SDVariable gain, boolean channelsFirst, long... dimensions) {
        this(sameDiff, input, gain, null, channelsFirst, dimensions);
    }

    public LayerNorm(INDArray input, INDArray gain, INDArray bias, INDArray result, boolean channelsFirst, long... dimensions) {
        super("layer_norm", wrapFilterNull(input, gain, bias), wrapOrNull(result));
        this.noBias = bias == null;
        this.channelsFirst = channelsFirst;
        setDimensions(dimensions);
    }

    public LayerNorm(@NonNull INDArray input, @NonNull INDArray gain, boolean channelsFirst, long... dimensions) {
        this(input, gain, null, channelsFirst, dimensions);
    }

    public LayerNorm(INDArray input, INDArray gain, INDArray result, boolean channelsFirst, long... dimensions) {
        this(input, gain, null, result, channelsFirst, dimensions);
    }

    @Override
    public void setDimensions(long[] dimensions) {
        Preconditions.checkArgument(dimensions != null, "LayerNorm: You have to provide dimensions");
        Preconditions.checkArgument(dimensions.length > 0, "LayerNorm: You have to provide dimensions");

        this.dimensions = dimensions;
        this.iArguments.clear();
        addIArgument(dimensions);
        this.bArguments.clear();
        this.bArguments.add(channelsFirst);
    }

    @Override
    public void addBArgument(boolean... arg) {
        super.addBArgument(arg);
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new HashMap<>();
        ret.put("noBias",noBias);
        ret.put("channelsFirst",channelsFirst);
        if(dimensions != null)
            ret.put("dimensions",dimensions);
        return ret;
    }

    @Override
    public void configureFromArguments() {
        if(!bArguments.isEmpty() && bArguments.size() > 1) {
            this.noBias = bArguments.get(1);
        }

        if(!bArguments.isEmpty()) {
            this.channelsFirst = bArguments.get(0);
        }

        if(!iArguments.isEmpty()) {
            this.dimensions = Longs.toArray(iArguments);
        }
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        Boolean noBias = getBooleanFromProperty("noBias",properties);
        if(noBias != null) {
            this.noBias = noBias;
        }

        Boolean channelsFirst = getBooleanFromProperty("channelsFirst",properties);
        if(channelsFirst != null) {
            this.channelsFirst = channelsFirst;
        }

        if(properties.containsKey("dimensions") && properties.get("dimensions") instanceof Long) {
            Long dimension = (Long) properties.get("dimensions");
            this.dimensions = new long[]{dimension.intValue()};
        } else if(properties.containsKey("dimensions") && properties.get("dimensions") instanceof int[]) {
            long[] dimensions = (long[]) properties.get("dimensions");
            this.dimensions = dimensions;
        }
    }

    @Override
    public String opName() {
        return "layer_norm";
    }


    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow name found for shape " + opName());
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradient) {
        if (noBias) {
            return new LayerNormBp(sameDiff, arg(0), arg(1), gradient.get(0), channelsFirst, dimensions).outputs();
        } else {
            return new LayerNormBp(sameDiff, arg(0), arg(1), arg(2), gradient.get(0), channelsFirst, dimensions).outputs();
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && dataTypes.size() >= 2 && dataTypes.size() <= 3, "Expected exactly 2 or 3 input datatypes, got %s", dataTypes);
        DataType first = dataTypes.get(0);


        return Collections.singletonList(first);
    }

    @Override
    public int numOutputArguments() {
        return noBias ? 2 : 3;
    }
}
