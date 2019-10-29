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
import org.nd4j.base.Preconditions;
import org.nd4j.jackson.objectmapper.holder.ObjectMapperHolder;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

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
    private String name =  UUID.randomUUID().toString();
    private Schema inputSchema;
    private Schema outputSchema;
    private String outputDict;
    private boolean returnAllVariables;


    @Builder
    public PythonTransform(String code,
                           PythonVariables inputs,
                           PythonVariables outputs,
                           String name,
                           Schema inputSchema,
                           Schema outputSchema,
                           String outputDict,
                           boolean returnAllInputs) {
        Preconditions.checkNotNull(code,"No code found to run!");
        this.code = code;
        this.returnAllVariables = returnAllInputs;
        if(inputs != null)
            this.inputs = inputs;
        if(outputs != null)
            this.outputs = outputs;

        if(name != null)
            this.name = name;
        if (outputDict != null) {
            this.outputDict = outputDict;
            this.outputs = new PythonVariables();
            this.outputs.addDict(outputDict);

            String helpers;
            try(InputStream is = new ClassPathResource("pythonexec/serialize_array.py").getInputStream()) {
                helpers = IOUtils.toString(is, Charset.defaultCharset());

            }catch (IOException e){
                throw  new RuntimeException("Error reading python code");
            }
            this.code += "\n\n" + helpers;
            this.code += "\n" + outputDict + " = __recursive_serialize_dict(" + outputDict + ")";
        }

        try {
            if(inputSchema != null) {
                this.inputSchema = inputSchema;
                if(inputs == null || inputs.isEmpty()) {
                    this.inputs = schemaToPythonVariables(inputSchema);
                }
            }

            if(outputSchema != null) {
                this.outputSchema = outputSchema;
                if(outputs == null || outputs.isEmpty()) {
                    this.outputs = schemaToPythonVariables(outputSchema);
                }
            }
        }catch(Exception e) {
            throw new IllegalStateException(e);
        }

    }


    @Override
    public void setInputSchema(Schema inputSchema) {
        Preconditions.checkNotNull(inputSchema,"No input schema found!");
        this.inputSchema = inputSchema;
        try{
            inputs = schemaToPythonVariables(inputSchema);
        }catch (Exception e){
            throw new RuntimeException(e);
        }
        if (outputSchema == null && outputDict == null){
            outputSchema = inputSchema;
        }

    }

    @Override
    public Schema getInputSchema(){
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
        Preconditions.checkNotNull(pyInputs,"Inputs must not be null!");


        try{
            if (returnAllVariables) {
                return getWritablesFromPyOutputs(PythonExecutioner.execAndReturnAllVariables(code, pyInputs));
            }

            if (outputDict != null) {
                PythonExecutioner.exec(this, pyInputs);
                PythonVariables out = PythonUtils.expandInnerDict(outputs, outputDict);
                return getWritablesFromPyOutputs(out);
            }
            else {
                PythonExecutioner.execWithSetupAndRun(code,pyInputs,outputs);
                return getWritablesFromPyOutputs(outputs);
            }

        }
        catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    @Override
    public String[] outputColumnNames(){
        return outputs.getVariables();
    }

    @Override
    public String outputColumnName(){
        return outputColumnNames()[0];
    }
    @Override
    public String[] columnNames(){
        return outputs.getVariables();
    }

    @Override
    public String columnName(){
        return columnNames()[0];
    }

    public Schema transform(Schema inputSchema){
        return outputSchema;
    }


    private PythonVariables getPyInputsFromWritables(List<Writable> writables) {
        PythonVariables ret = new PythonVariables();

        for (String name: inputs.getVariables()) {
            int colIdx = inputSchema.getIndexOfColumn(name);
            Writable w = writables.get(colIdx);
            PythonVariables.Type pyType = inputs.getType(name);
            switch (pyType){
                case INT:
                    if (w instanceof LongWritable){
                        ret.addInt(name, ((LongWritable)w).get());
                    }
                    else{
                        ret.addInt(name, ((IntWritable)w).get());
                    }

                    break;
                case FLOAT:
                    if (w instanceof DoubleWritable) {
                        ret.addFloat(name, ((DoubleWritable)w).get());
                    }
                    else{
                        ret.addFloat(name, ((FloatWritable)w).get());
                    }
                    break;
                case STR:
                    ret.addStr(name, w.toString());
                    break;
                case NDARRAY:
                    ret.addNDArray(name,((NDArrayWritable)w).get());
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
            PythonVariables.Type pyType = pyOuts.getType(name);
            switch (pyType){
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
                    NumpyArray arr = pyOuts.getNDArrayValue(name);
                    schemaBuilder.addColumnNDArray(name, arr.getShape());
                    break;
                default:
                    throw new IllegalStateException("Unable to support type " + pyType.name());
            }
        }
        this.outputSchema = schemaBuilder.build();


        for (int i = 0; i < varNames.length; i++) {
            String name = varNames[i];
            PythonVariables.Type pyType = pyOuts.getType(name);

            switch (pyType){
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
                    NumpyArray arr = pyOuts.getNDArrayValue(name);
                    out.add(new NDArrayWritable(arr.getNd4jArray()));
                    break;
                case DICT:
                    Map<?, ?> dictValue = pyOuts.getDictValue(name);
                    try {
                        out.add(new Text(ObjectMapperHolder.getJsonMapper().writeValueAsString(dictValue)));
                    } catch (JsonProcessingException e) {
                        throw new IllegalStateException("Unable to serialize dictionary " + name + " to json!");
                    }
                    break;
                case LIST:
                    Object[] listValue = pyOuts.getListValue(name);
                    try {
                        out.add(new Text(ObjectMapperHolder.getJsonMapper().writeValueAsString(listValue)));
                    } catch (JsonProcessingException e) {
                        throw new IllegalStateException("Unable to serialize list vlaue " + name + " to json!");
                    }
                    break;
                    /*
                    case DICT:
                    Map<?,?> outMap = pyOuts.getDictValue(name);
                    for(val entry : outMap.entrySet()) {
                    addPrimitiveWritable(out,entry.getValue());
                    }
                    break;
                    */
                default:
                    throw new IllegalStateException("Unable to support type " + pyType.name());
            }
        }
        return out;
    }



    private PythonVariables schemaToPythonVariables(Schema schema) throws Exception {
        PythonVariables pyVars = new PythonVariables();
        int numCols = schema.numColumns();
        for (int i = 0; i < numCols; i++) {
            String colName = schema.getName(i);
            ColumnType colType = schema.getType(i);
            switch (colType){
                case Long:
                case Integer:
                    pyVars.addInt(colName);
                    break;
                case Double:
                case Float:
                    pyVars.addFloat(colName);
                    break;
                case String:
                    pyVars.addStr(colName);
                    break;
                case NDArray:
                    pyVars.addNDArray(colName);
                    break;
                default:
                    throw new Exception("Unsupported python input type: " + colType.toString());
            }
        }

        return pyVars;
    }




}