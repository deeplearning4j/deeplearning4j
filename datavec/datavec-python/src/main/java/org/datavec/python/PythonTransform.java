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

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.io.IOUtils;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;
import org.json.JSONPropertyIgnore;
import org.nd4j.base.Preconditions;
import org.nd4j.jackson.objectmapper.holder.ObjectMapperHolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.datavec.python.PythonUtils.schemaToPythonVariables;

/**
 * Row-wise Transform that applies arbitrary python code on each row
 *
 * @author Fariz Rahman
 */

@NoArgsConstructor
@Data
public class PythonTransform implements Transform {

    private String code;
    private PythonVariables inputs;
    private PythonVariables outputs;
    private String name = UUID.randomUUID().toString();
    private Schema inputSchema;
    private Schema outputSchema;
    private String outputDict;
    private boolean returnAllVariables;
    private boolean setupAndRun = false;
    private PythonJob pythonJob;


    @Builder
    public PythonTransform(String code,
                           PythonVariables inputs,
                           PythonVariables outputs,
                           String name,
                           Schema inputSchema,
                           Schema outputSchema,
                           String outputDict,
                           boolean returnAllInputs,
                           boolean setupAndRun) {
        Preconditions.checkNotNull(code, "No code found to run!");
        this.code = code;
        this.returnAllVariables = returnAllInputs;
        this.setupAndRun = setupAndRun;
        if (inputs != null)
            this.inputs = inputs;
        if (outputs != null)
            this.outputs = outputs;
        if (name != null)
            this.name = name;
        if (outputDict != null) {
            this.outputDict = outputDict;
            this.outputs = new PythonVariables();
            this.outputs.addDict(outputDict);
        }

        try {
            if (inputSchema != null) {
                this.inputSchema = inputSchema;
                if (inputs == null || inputs.isEmpty()) {
                    this.inputs = schemaToPythonVariables(inputSchema);
                }
            }

            if (outputSchema != null) {
                this.outputSchema = outputSchema;
                if (outputs == null || outputs.isEmpty()) {
                    this.outputs = schemaToPythonVariables(outputSchema);
                }
            }
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        try{
            pythonJob = PythonJob.builder()
                    .name("a" + UUID.randomUUID().toString().replace("-", "_"))
                    .code(code)
                    .setupRunMode(setupAndRun)
                    .build();
        }
        catch(Exception e){
            throw new IllegalStateException("Error creating python job: " + e);
        }

    }


    @Override
    public void setInputSchema(Schema inputSchema) {
        Preconditions.checkNotNull(inputSchema, "No input schema found!");
        this.inputSchema = inputSchema;
        try {
            inputs = schemaToPythonVariables(inputSchema);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        if (outputSchema == null && outputDict == null) {
            outputSchema = inputSchema;
        }

    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        List<List<Writable>> out = new ArrayList<>();
        for (List<Writable> l : sequence) {
            out.add(map(l));
        }
        return out;
    }

    @Override
    public Object map(Object input) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Object mapSequence(Object sequence) {
        throw new UnsupportedOperationException("Not yet implemented");
    }


    @Override
    public List<Writable> map(List<Writable> writables) {
        PythonVariables pyInputs = getPyInputsFromWritables(writables);
        Preconditions.checkNotNull(pyInputs, "Inputs must not be null!");
        try {
            if (returnAllVariables) {
                return getWritablesFromPyOutputs(pythonJob.execAndReturnAllVariables(pyInputs));
            }

            if (outputDict != null) {
                pythonJob.exec(pyInputs, outputs);
                PythonVariables out = PythonUtils.expandInnerDict(outputs, outputDict);
                return getWritablesFromPyOutputs(out);
            } else {
                pythonJob.exec(pyInputs, outputs);

                return getWritablesFromPyOutputs(outputs);
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String[] outputColumnNames() {
        return outputs.getVariables();
    }

    @Override
    public String outputColumnName() {
        return outputColumnNames()[0];
    }

    @Override
    public String[] columnNames() {
        return outputs.getVariables();
    }

    @Override
    public String columnName() {
        return columnNames()[0];
    }

    public Schema transform(Schema inputSchema) {
        return outputSchema;
    }


    private PythonVariables getPyInputsFromWritables(List<Writable> writables) {
        PythonVariables ret = new PythonVariables();

        for (String name : inputs.getVariables()) {
            int colIdx = inputSchema.getIndexOfColumn(name);
            Writable w = writables.get(colIdx);
            PythonType pyType = inputs.getType(name);
            switch (pyType.getName()) {
                case INT:
                    if (w instanceof LongWritable) {
                        ret.addInt(name, ((LongWritable) w).get());
                    } else {
                        ret.addInt(name, ((IntWritable) w).get());
                    }
                    break;
                case FLOAT:
                    if (w instanceof DoubleWritable) {
                        ret.addFloat(name, ((DoubleWritable) w).get());
                    } else {
                        ret.addFloat(name, ((FloatWritable) w).get());
                    }
                    break;
                case STR:
                    ret.addStr(name, w.toString());
                    break;
                case NDARRAY:
                    ret.addNDArray(name, ((NDArrayWritable) w).get());
                    break;
                case BOOL:
                    ret.addBool(name, ((BooleanWritable) w).get());
                    break;
                default:
                    throw new RuntimeException("Unsupported input type:" + pyType);
            }

        }
        return ret;
    }

    private List<Writable> getWritablesFromPyOutputs(PythonVariables pyOuts) {
        List<Writable> out = new ArrayList<>();
        String[] varNames;
        varNames = pyOuts.getVariables();
        Schema.Builder schemaBuilder = new Schema.Builder();
        for (int i = 0; i < varNames.length; i++) {
            String name = varNames[i];
            PythonType pyType = pyOuts.getType(name);
            switch (pyType.getName()) {
                case INT:
                    schemaBuilder.addColumnLong(name);
                    break;
                case FLOAT:
                    schemaBuilder.addColumnDouble(name);
                    break;
                case STR:
                case DICT:
                case LIST:
                    schemaBuilder.addColumnString(name);
                    break;
                case NDARRAY:
                    INDArray arr = pyOuts.getNDArrayValue(name);
                    schemaBuilder.addColumnNDArray(name, arr.shape());
                    break;
                case BOOL:
                    schemaBuilder.addColumnBoolean(name);
                    break;
                default:
                    throw new IllegalStateException("Unable to support type " + pyType.getName());
            }
        }
        this.outputSchema = schemaBuilder.build();


        for (int i = 0; i < varNames.length; i++) {
            String name = varNames[i];
            PythonType pyType = pyOuts.getType(name);

            switch (pyType.getName()) {
                case INT:
                    out.add(new LongWritable(pyOuts.getIntValue(name)));
                    break;
                case FLOAT:
                    out.add(new DoubleWritable(pyOuts.getFloatValue(name)));
                    break;
                case STR:
                    out.add(new Text(pyOuts.getStrValue(name)));
                    break;
                case NDARRAY:
                    INDArray arr = pyOuts.getNDArrayValue(name);
                    out.add(new NDArrayWritable(arr));
                    break;
                case DICT:
                    Map<?, ?> dictValue = pyOuts.getDictValue(name);
                    Map noNullValues = new java.util.HashMap<>();
                    for (Map.Entry entry : dictValue.entrySet()) {
                        if (entry.getValue() != org.json.JSONObject.NULL) {
                            noNullValues.put(entry.getKey(), entry.getValue());
                        }
                    }

                    try {
                        out.add(new Text(ObjectMapperHolder.getJsonMapper().writeValueAsString(noNullValues)));
                    } catch (JsonProcessingException e) {
                        throw new IllegalStateException("Unable to serialize dictionary " + name + " to json!");
                    }
                    break;
                case LIST:
                    Object[] listValue = pyOuts.getListValue(name).toArray();
                    try {
                        out.add(new Text(ObjectMapperHolder.getJsonMapper().writeValueAsString(listValue)));
                    } catch (JsonProcessingException e) {
                        throw new IllegalStateException("Unable to serialize list vlaue " + name + " to json!");
                    }
                    break;
                case BOOL:
                    out.add(new BooleanWritable(pyOuts.getBooleanValue(name)));
                    break;
                default:
                    throw new IllegalStateException("Unable to support type " + pyType.getName());
            }
        }
        return out;
    }


}