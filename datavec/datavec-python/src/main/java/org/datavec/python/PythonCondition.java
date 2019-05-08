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

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;

import java.util.List;

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


    public PythonCondition(String pythonCode){
        code = pythonCode;
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
                    ret.addFloat(name, ((DoubleWritable)w).get());
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
    @Override
    public void setInputSchema(Schema inputSchema){
        this.inputSchema = inputSchema;
        try{
            pyInputs = schemaToPythonVariables(inputSchema);
            PythonVariables pyOuts = new PythonVariables();
            pyOuts.addInt("out");
            pythonTransform = new PythonTransform(
                    code + "\n\nout=f()\nout=0 if out is None else int(out)", // TODO: remove int conversion after boolean support is covered
                    pyInputs,
                    pyOuts
            );
        }
        catch (Exception e){
            throw new RuntimeException(e);
        }



    }

    @Override
    public Schema getInputSchema(){
        return inputSchema;
    }

    public String[] outputColumnNames(){
        String[] columnNames = new String[inputSchema.numColumns()];
        inputSchema.getColumnNames().toArray(columnNames);
        return columnNames;
    }

    public String outputColumnName(){
        return outputColumnNames()[0];
    }

    public String[] columnNames(){
        return outputColumnNames();
    }

    public String columnName(){
        return outputColumnName();
    }

    public Schema transform(Schema inputSchema){
        return inputSchema;
    }

    public boolean condition(List<Writable> list){
        PythonVariables inputs = getPyInputsFromWritables(list);
        try{
            PythonExecutioner.exec(pythonTransform.getCode(), inputs, pythonTransform.getOutputs());
            boolean ret = pythonTransform.getOutputs().getIntValue("out") != 0;
            return ret;
        }
        catch (Exception e){
            throw new RuntimeException(e);
        }

    }

    public boolean condition(Object input){
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


}
