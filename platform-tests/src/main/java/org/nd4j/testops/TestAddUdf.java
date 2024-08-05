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
package org.nd4j.testops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.UserDefinedCustomOp;
import org.nd4j.linalg.api.ops.UserDefinedOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.AddBpOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@UserDefinedOp
public class TestAddUdf extends UserDefinedCustomOp {
    public TestAddUdf() {
        super();
    }

    public TestAddUdf(SameDiff sameDiff, SDVariable arg) {
        super(sameDiff, arg);
    }

    public TestAddUdf(SameDiff sameDiff, SDVariable[] args) {
        super(sameDiff, args);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Arrays.asList(dataTypes.get(0));
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {

    }

    @Override
    public Object getValue(Field property) {
        return null;
    }

    @Override
    public void setValueFor(Field target, Object value) {

    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return Collections.emptyMap();
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }

    @Override
    public String opName() {
        return "test_add_udf";
    }

    @Override
    public void configureFromArguments() {

    }

    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    @Override
    public boolean isInplaceCall() {
        return false;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        return Arrays.asList(inputArguments.get(0).shapeDescriptor());
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(OpContext oc) {
        return Arrays.asList(oc.getInputArrays().get(0).shapeDescriptor());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return new AddBpOp(sameDiff, larg(), rarg(), f1.get(0)).outputs();
    }

    @Override
    public void exec() {
        AddOp addOp = new AddOp();
        addOp.addInputArgument(inputArguments.get(0),inputArguments.get(1));
        Nd4j.getExecutioner().exec(addOp);
        this.outputArguments.addAll(addOp.outputArguments());
    }

    @Override
    public void exec(OpContext opContext) {
        Nd4j.getExecutioner().exec(new AddOp(),opContext);
    }
}
