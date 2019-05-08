/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Row-wise Transform that applies arbitrary python code on each row
 *
 * @author Fariz Rahman
 */

@NoArgsConstructor
@Data
public class PythonTransform implements Transform{
    private String code;
    private PythonVariables pyInputs;
    private PythonVariables pyOutputs;
    private String name;
    private Schema inputSchema;
    private Schema outputSchema;


    public PythonTransform(String code, PythonVariables pyInputs, PythonVariables pyOutputs) throws Exception{
        this.code = code;
        this.pyInputs = pyInputs;
        this.pyOutputs = pyOutputs;
        this.name = UUID.randomUUID().toString();
    }

    @Override
    public void setInputSchema(Schema inputSchema){
        this.inputSchema = inputSchema;
        try{
            pyInputs = schemaToPythonVariables(inputSchema);
        }catch (Exception e){
            throw new RuntimeException(e);
        }
        if (outputSchema == null){
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
    public List<Writable> map(List<Writable> writables){
        PythonVariables pyInputs = getPyInputsFromWritables(writables);
        try{
            PythonExecutioner.exec(code, pyInputs, pyOutputs);
            return getWritablesFromPyOutputs(pyOutputs);
        }
        catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    @Override
    public String[] outputColumnNames(){
        return pyOutputs.getVariables();
    }

    @Override
    public String outputColumnName(){
        return outputColumnNames()[0];
    }
    @Override
    public String[] columnNames(){
        return pyOutputs.getVariables();
    }

    @Override
    public String columnName(){
        return columnNames()[0];
    }

    public Schema transform(Schema inputSchema){
        return outputSchema;
    }


    private PythonVariables getPyInputsFromWritables(List<Writable> writables){

        PythonVariables ret = new PythonVariables();

        for (String name: pyInputs.getVariables()){
            int colIdx = inputSchema.getIndexOfColumn(name);
            Writable w = writables.get(colIdx);
            PythonVariables.Type pyType = pyInputs.getType(name);
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
                    if (w instanceof DoubleWritable){
                        ret.addFloat(name, ((DoubleWritable)w).get());
                    }
                    else{
                        ret.addFloat(name, ((FloatWritable)w).get());
                    }
                    break;
                case STR:
                    ret.addStr(name, ((Text)w).toString());
                    break;
                case NDARRAY:
                    ret.addNDArray(name,((NDArrayWritable)w).get());
                    break;
            }

        }
        return ret;
    }

    private List<Writable> getWritablesFromPyOutputs(PythonVariables pyOuts){
        List<Writable> out = new ArrayList<>();
        for (int i=0; i<outputSchema.numColumns(); i++){
            String name = outputSchema.getName(i);
            PythonVariables.Type pyType = pyOutputs.getType(name);
            switch (pyType){
                case INT:
                    out.add((Writable) new LongWritable(pyOuts.getIntValue(name)));
                    break;
                case FLOAT:
                    out.add((Writable) new DoubleWritable(pyOuts.getFloatValue(name)));
                    break;
                case STR:
                    out.add((Writable) new Text(pyOuts.getStrValue(name)));
                    break;
                case NDARRAY:
                    out.add((Writable) new NDArrayWritable(pyOuts.getNDArrayValue(name).getNd4jArray()));
                    break;
            }
        }
        return out;
    }


    public PythonTransform(String code) throws Exception{
        this.code = code;
        this.name = UUID.randomUUID().toString();
    }
    private PythonVariables schemaToPythonVariables(Schema schema) throws Exception{
        PythonVariables pyVars = new PythonVariables();
        int numCols = schema.numColumns();
        for (int i=0; i<numCols; i++){
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

    public PythonTransform(String code, Schema outputSchema) throws Exception{
        this.code = code;
        this.name = UUID.randomUUID().toString();
        this.outputSchema = outputSchema;
        this.pyOutputs = schemaToPythonVariables(outputSchema);


    }
    public String getName() {
        return name;
    }

    public String getCode(){
        return code;
    }

    public PythonVariables getInputs() {
        return pyInputs;
    }

    public PythonVariables getOutputs() {
        return pyOutputs;
    }


}
