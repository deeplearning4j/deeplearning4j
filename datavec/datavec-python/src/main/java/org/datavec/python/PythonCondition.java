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

package org.datavec.python;

import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;
import java.util.List;

import static org.datavec.python.PythonUtils.schemaToPythonVariables;
import static org.nd4j.common.base.Preconditions.checkNotNull;
import static org.nd4j.common.base.Preconditions.checkState;

/**
 * Lets a condition be defined as a python method f that takes no arguments
 * and returns a boolean indicating whether or not to filter a row.
 * The values of all columns in current row are available as global variables to f.
 *
 * @author Fariz Rahman
 */
public class PythonCondition implements Condition {

    private Schema inputSchema;
    private PythonVariables pyInputs;
    private PythonTransform pythonTransform;
    private String code;


    public PythonCondition(String pythonCode) {
        checkNotNull("Python code must not be null!", pythonCode);
        checkState(!pythonCode.isEmpty(), "Python code must not be empty!");
        code = pythonCode;
    }


    @Override
    public void setInputSchema(Schema inputSchema) {
        this.inputSchema = inputSchema;
        try {
            pyInputs = schemaToPythonVariables(inputSchema);
            PythonVariables pyOuts = new PythonVariables();
            pyOuts.addInt("out");
            pythonTransform = PythonTransform.builder()
                    .code(code + "\n\nout=f()\nout=0 if out is None else int(out)")
                    .inputs(pyInputs)
                    .outputs(pyOuts)
                    .build();

        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public String[] outputColumnNames() {
        String[] columnNames = new String[inputSchema.numColumns()];
        inputSchema.getColumnNames().toArray(columnNames);
        return columnNames;
    }

    @Override
    public String outputColumnName() {
        return outputColumnNames()[0];
    }

    @Override
    public String[] columnNames() {
        return outputColumnNames();
    }

    @Override
    public String columnName() {
        return outputColumnName();
    }

    @Override
    public Schema transform(Schema inputSchema) {
        return inputSchema;
    }

    @Override
    public boolean condition(List<Writable> list) {
        PythonVariables inputs = getPyInputsFromWritables(list);
        try {
            pythonTransform.getPythonJob().exec(inputs, pythonTransform.getOutputs());
            boolean ret = pythonTransform.getOutputs().getIntValue("out") != 0;
            return ret;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean condition(Object input) {
        return condition(input);
    }

    @Override
    public boolean conditionSequence(List<List<Writable>> list) {
        throw new UnsupportedOperationException("not supported");
    }


    @Override
    public boolean conditionSequence(Object input) {
        throw new UnsupportedOperationException("not supported");
    }

    private PythonVariables getPyInputsFromWritables(List<Writable> writables) {
        PythonVariables ret = new PythonVariables();

        for (int i = 0; i < inputSchema.numColumns(); i++) {
            String name = inputSchema.getName(i);
            Writable w = writables.get(i);
            PythonType pyType = pyInputs.getType(inputSchema.getName(i));
            switch (pyType.getName()) {
                case INT:
                    if (w instanceof LongWritable) {
                        ret.addInt(name, ((LongWritable) w).get());
                    } else {
                        ret.addInt(name, ((IntWritable) w).get());
                    }

                    break;
                case FLOAT:
                    ret.addFloat(name, ((DoubleWritable) w).get());
                    break;
                case STR:
                    ret.addStr(name, w.toString());
                    break;
                case NDARRAY:
                    ret.addNDArray(name, ((NDArrayWritable) w).get());
                    break;
            }
        }

        return ret;
    }


}