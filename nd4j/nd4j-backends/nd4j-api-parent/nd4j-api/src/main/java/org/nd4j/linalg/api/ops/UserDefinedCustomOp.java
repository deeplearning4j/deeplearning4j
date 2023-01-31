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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;

public abstract class UserDefinedCustomOp extends DynamicCustomOp {

    public UserDefinedCustomOp() {
        super();
    }

    public UserDefinedCustomOp(SameDiff sameDiff, SDVariable arg) {
        super(sameDiff, arg);
    }

    public UserDefinedCustomOp(SameDiff sameDiff, SDVariable[] args) {
        super(sameDiff, args);
    }

    public UserDefinedCustomOp(String opName, SameDiff sameDiff, SDVariable[] args) {
        super(opName, sameDiff, args);
    }

    public UserDefinedCustomOp(String opName, INDArray input, INDArray output, List<Double> tArguments, int[] iArguments) {
        super(opName, input, output, tArguments, iArguments);
    }

    public UserDefinedCustomOp(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, int[] iArguments) {
        super(opName, inputs, outputs, tArguments, iArguments);
    }

    public UserDefinedCustomOp(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments) {
        super(opName, inputs, outputs, tArguments, iArguments);
    }

    public UserDefinedCustomOp(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    public UserDefinedCustomOp(String opName, INDArray[] inputs, INDArray[] outputs) {
        super(opName, inputs, outputs);
    }

    public UserDefinedCustomOp(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(opName, sameDiff, args, inPlace);
    }

    public UserDefinedCustomOp(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(sameDiff, args, inPlace);
    }

    protected UserDefinedCustomOp(String opName) {
        super(opName);
    }

    @Override
    public Op.Type opType() {
        return Op.Type.UDF;
    }

    @Override
    public abstract List<DataType> calculateOutputDataTypes(List<DataType> dataTypes);

    @Override
    public abstract void setPropertiesForFunction(Map<String, Object> properties);
    @Override
    public abstract  Object getValue(Field property);

    @Override
    public abstract  void setValueFor(Field target, Object value);

    @Override
    public abstract  Map<String, Object> propertiesForFunction();

    @Override
    public long opHash() {
        return 1;
    }

    @Override
    public abstract int getNumOutputs();
    @Override
    public abstract String opName();

    @Override
    public abstract void configureFromArguments();

    @Override
    public abstract void configureWithSameDiff(SameDiff sameDiff);

    @Override
    public abstract boolean isInplaceCall();
    @Override
    public abstract List<LongShapeDescriptor> calculateOutputShape();

    @Override
    public abstract List<LongShapeDescriptor> calculateOutputShape(OpContext oc);
    @Override
    public abstract  List<SDVariable> doDiff(List<SDVariable> f1);

    public abstract void exec();

    public abstract void exec(OpContext opContext);
}
