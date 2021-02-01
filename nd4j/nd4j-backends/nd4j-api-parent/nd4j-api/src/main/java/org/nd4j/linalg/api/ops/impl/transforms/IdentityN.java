/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.Collections;
import java.util.List;

/**
 * IdentityN op wrapper
 *
 * @author raver119@gmail.com
 */
public class IdentityN extends DynamicCustomOp {

    public IdentityN() {}

    public IdentityN(SameDiff sameDiff, SDVariable[] inputs) {
        super(null, sameDiff, inputs, false);
    }

    public IdentityN(SameDiff sameDiff, SDVariable input) {
        super(null, sameDiff, new SDVariable[] {input}, false);
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }

    @Override
    public String tensorflowName() {
        return "IdentityN";
    }

    @Override
    public String opName() {
        return "identity_n";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public int getNumOutputs(){
        return args().length;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> list){
        return list;
    }
}
