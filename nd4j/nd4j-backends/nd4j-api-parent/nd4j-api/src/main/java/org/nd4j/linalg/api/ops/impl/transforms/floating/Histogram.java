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

package org.nd4j.linalg.api.ops.impl.transforms.floating;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformFloatOp;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
public class Histogram extends BaseTransformFloatOp {
    public Histogram(SameDiff sameDiff, SDVariable i_v, boolean inPlace, int numBins) {
        super(sameDiff, i_v, inPlace);
        this.numBins = numBins;
    }

    public Histogram(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs, int numBins) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
        this.numBins = numBins;
    }

    public Histogram(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs, int numBins) {
        super(sameDiff, i_v, extraArgs);
        this.numBins = numBins;
    }

    private int numBins = 0;

    public Histogram() {
        //no-op
    }

    public Histogram(INDArray x, INDArray z) {
        setX(x);
        setZ(z);

        //FIXME: int cast
        numBins = (int) z.length();

        double max = x.maxNumber().doubleValue();
        double min = x.minNumber().doubleValue();

        this.extraArgs = new Object[] {(double) numBins, min, max};
    }

    public Histogram(INDArray x, int numberOfBins) {
        this(x, Nd4j.create(numberOfBins));
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("numBins",numBins);
        return ret;
    }

    @Override
    public int opNum() {
        return 48;
    }

    @Override
    public String opName() {
        return "histogram";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Not supported");
    }
}
