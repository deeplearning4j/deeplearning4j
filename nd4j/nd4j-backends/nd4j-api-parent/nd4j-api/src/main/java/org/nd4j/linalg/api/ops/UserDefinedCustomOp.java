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

/**
 * A mandatory base class for creating UDFs.
 * UDFs are custom ops created by a user to handle
 * creating custom gradients while being properly integrated
 * with samediff.
 *
 * Users should override:
 * 1. {@link #exec()} for execution with plain input arrays. A user may
 * add anything they wish in here. Just ensure that when done
 * {@link #outputArguments} contains the final outputs to match the
 * expected op outputs.
 * 2. {@link #exec(OpContext)} for execution with op contexts. Same as above
 * but please use {@link OpContext#getOutputArrays()} add for the final results.
 *
 *
 *  Lastly, for metadata purposes, users need to override every method
 * like providing an op name and any properties needed by the op
 * when being instantiated.
 * In terms of serialization, a user's UDF should have;
 * 1. an empty constructor. This is used when creating a graph from flatbuffers in the underlying {@link org.nd4j.autodiff.samediff.serde.FlatBuffersMapper}.
 * 2. {@link #configureWithSameDiff(SameDiff)} implemented: this is for handling initialization after
 * the op is created. This will initiate values using the relevant samediff metadata. This includes obtaining things like
 * input and output argument metadata from {@link SDVariable} found as {@link #args()}
 * 3. {@link #configureFromArguments()} for configuration from specified arguments such as ints, floats/doubles, and input variables.
 * The arguments referenced are the underlying arguments that get passed to every c/c++ ops. This includes
 * the {@link #iArguments} {@link #tArguments} {@link #dArguments} {@link #inputArguments}
 * {@link #outputArguments}
 *
 *
 * 4. A user can define properties as fields. if a user does so,
 * please ensure that you implement {@link #setPropertiesForFunction(Map)}
 * {@link #propertiesForFunction()} these are used to create an op from scratch
 * when saving/loading a model.
 *
 * 5. A user must implement {@link #calculateOutputDataTypes(List)} this is used in
 * samediff to determine how many output variables are needed when it can't determine that
 * from {@link #getNumOutputs()}
 *
 * 6. A user must implement {@link #doDiff(List)} this is where a user's custom gradient definition goes.
 *
 *
 * @author Adam Gibson
 */
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

    /**
     * Override this method for execution.
     */
    public abstract void exec();

    /**
     * Override this method for execution with an op context.
     * @param opContext
     */
    public abstract void exec(OpContext opContext);

    @Override
    public boolean equals(Object o) {
        if(!o.getClass().equals(getClass()))
            return false;
        UserDefinedCustomOp userDefinedCustomOp = (UserDefinedCustomOp) o;
        return opType() == userDefinedCustomOp.opType() && opName().equals(userDefinedCustomOp.opName()) &&
                getOwnName().equals(userDefinedCustomOp.getOwnName());
    }
}
