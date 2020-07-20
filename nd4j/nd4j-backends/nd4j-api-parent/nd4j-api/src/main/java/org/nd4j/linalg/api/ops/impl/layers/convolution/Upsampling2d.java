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

package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;


/**
 * Upsampling operation
 */
@Slf4j
@Getter
@NoArgsConstructor
public class Upsampling2d extends DynamicCustomOp {


    protected boolean nchw;
    protected int scaleH;
    protected int scaleW;

    public Upsampling2d(SameDiff sameDiff, SDVariable input, boolean nchw, int scaleH, int scaleW) {
        super(null,sameDiff, new SDVariable[]{input});
        this.nchw = nchw;
        this.scaleH = scaleH;
        this.scaleW = scaleW;

        addIArgument(scaleH);
        addIArgument(scaleW);
        addIArgument(nchw ? 1 : 0);
    }

    public Upsampling2d(SameDiff sameDiff, SDVariable input, int scaleH, int scaleW, boolean nchw) {
        this(sameDiff, input, nchw, scaleH, scaleW);
    }

    public Upsampling2d(SameDiff sameDiff, SDVariable input, int scale) {
        super(null,sameDiff, new SDVariable[]{input});
        addIArgument(scale);
    }

    public Upsampling2d(INDArray input, int scale) {
        this(input, scale, scale, true);
    }

    public Upsampling2d(INDArray input, int scaleH, int scaleW, boolean nchw) {
        super(new INDArray[]{input}, null);
        this.nchw = nchw;
        this.scaleH = scaleH;
        this.scaleW = scaleW;

        addIArgument(scaleH);
        addIArgument(scaleW);
        addIArgument(nchw ? 1 : 0);
    }


    @Override
    public String opName() {
        return "upsampling2d";
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
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return new Upsampling2dDerivative(sameDiff, arg(), f1.get(0), nchw, scaleH, scaleW).outputs();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1, "Expected 1 input data type for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
