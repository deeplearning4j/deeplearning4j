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

import lombok.Data;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.*;

/**
 * Holds python variable names, types and values.
 * Also handles mapping from java types to python types.
 *
 * @author Fariz Rahman
 */

@Data
public class PythonVariables implements Serializable {

    public enum Type{
        BOOL,
        STR,
        INT,
        FLOAT,
        NDARRAY,
        LIST,
        FILE,
        DICT

    }

    private Map<String, String> strVariables = new LinkedHashMap<>();
    private Map<String, Long> intVariables = new LinkedHashMap<>();
    private Map<String, Double> floatVariables = new LinkedHashMap<>();
    private Map<String, Boolean> boolVariables = new LinkedHashMap<>();
    private Map<String, NumpyArray> ndVars = new LinkedHashMap<>();
    private Map<String, Object[]> listVariables = new LinkedHashMap<>();
    private Map<String, String> fileVariables = new LinkedHashMap<>();
    private Map<String,Map<?,?>> dictVariables = new LinkedHashMap<>();
    private Map<String, Type> vars = new LinkedHashMap<>();
    private Map<Type, Map> maps = new LinkedHashMap<>();


    /**
     * Returns a copy of the variable
     * schema in this array without the values
     * @return an empty variables clone
     * with no values
     */
    public PythonVariables copySchema(){
        PythonVariables ret = new PythonVariables();
        for (String varName: getVariables()){
            Type type = getType(varName);
            ret.add(varName, type);
        }
        return ret;
    }

    /**
     *
     */
    public PythonVariables() {
        maps.put(Type.BOOL, boolVariables);
        maps.put(Type.STR, strVariables);
        maps.put(Type.INT, intVariables);
        maps.put(Type.FLOAT, floatVariables);
        maps.put(Type.NDARRAY, ndVars);
        maps.put(Type.LIST, listVariables);
        maps.put(Type.FILE, fileVariables);
        maps.put(Type.DICT, dictVariables);

    }



    /**
     *
     * @return true if there are no variables.
     */
    public boolean isEmpty() {
        return getVariables().length < 1;
    }


    /**
     *
     * @param name Name of the variable
     * @param type Type of the variable
     */
    public void add(String name, Type type){
        switch (type){
            case BOOL:
                addBool(name);
                break;
            case STR:
                addStr(name);
                break;
            case INT:
                addInt(name);
                break;
            case FLOAT:
                addFloat(name);
                break;
            case NDARRAY:
                addNDArray(name);
                break;
            case LIST:
                addList(name);
                break;
            case FILE:
                addFile(name);
                break;
            case DICT:
                addDict(name);
        }
    }

    /**
     *
     * @param name name of the variable
     * @param type type of the variable
     * @param value value of the variable (must be instance of expected type)
     */
    public void add(String name, Type type, Object value) {
        add(name, type);
        setValue(name, value);
    }


    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addDict(String name) {
        vars.put(name, Type.DICT);
        dictVariables.put(name,null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addBool(String name){
        vars.put(name, Type.BOOL);
        boolVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addStr(String name){
        vars.put(name, Type.STR);
        strVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addInt(String name){
        vars.put(name, Type.INT);
        intVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addFloat(String name){
        vars.put(name, Type.FLOAT);
        floatVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addNDArray(String name){
        vars.put(name, Type.NDARRAY);
        ndVars.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addList(String name){
        vars.put(name, Type.LIST);
        listVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addFile(String name){
        vars.put(name, Type.FILE);
        fileVariables.put(name, null);
    }

    /**
     * Add a boolean variable to
     * the set of variables
     * @param name the field to add
     * @param value the value to add
     */
    public void addBool(String name, boolean value) {
        vars.put(name, Type.BOOL);
        boolVariables.put(name, value);
    }

    /**
     * Add a string variable to
     * the set of variables
     * @param name the field to add
     * @param value the value to add
     */
    public void addStr(String name, String value) {
        vars.put(name, Type.STR);
        strVariables.put(name, value);
    }

    /**
     * Add an int variable to
     * the set of variables
     * @param name the field to add
     * @param value the value to add
     */
    public void addInt(String name, int value) {
        vars.put(name, Type.INT);
        intVariables.put(name, (long)value);
    }

    /**
     * Add a long variable to
     * the set of variables
     * @param name the field to add
     * @param value the value to add
     */
    public void addInt(String name, long value) {
        vars.put(name, Type.INT);
        intVariables.put(name, value);
    }

    /**
     * Add a double variable to
     * the set of variables
     * @param name the field to add
     * @param value the value to add
     */
    public void addFloat(String name, double value) {
        vars.put(name, Type.FLOAT);
        floatVariables.put(name, value);
    }

    /**
     * Add a float variable to
     * the set of variables
     * @param name the field to add
     * @param value the value to add
     */
    public void addFloat(String name, float value) {
        vars.put(name, Type.FLOAT);
        floatVariables.put(name, (double)value);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     * @param value the value to add
     */
    public void addNDArray(String name, NumpyArray value) {
        vars.put(name, Type.NDARRAY);
        ndVars.put(name, value);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     * @param value the value to add
     */
    public void addNDArray(String name, INDArray value) {
        vars.put(name, Type.NDARRAY);
        ndVars.put(name, new NumpyArray(value));
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     * @param value the value to add
     */
    public void addList(String name, Object[] value) {
        vars.put(name, Type.LIST);
        listVariables.put(name, value);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     * @param value the value to add
     */
    public void addFile(String name, String value) {
        vars.put(name, Type.FILE);
        fileVariables.put(name, value);
    }


    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     * @param value the value to add
     */
    public void addDict(String name, Map value) {
        vars.put(name, Type.DICT);
        dictVariables.put(name, value);
    }
    /**
     *
     * @param name name of the variable
     * @param value new value for the variable
     */
    public void setValue(String name, Object value) {
        Type type = vars.get(name);
        if (type == Type.BOOL){
            boolVariables.put(name, (Boolean)value);
        }
        else if (type == Type.INT){
            Number number = (Number) value;
            intVariables.put(name, number.longValue());
        }
        else if (type == Type.FLOAT){
            Number number = (Number) value;
            floatVariables.put(name, number.doubleValue());
        }
        else if (type == Type.NDARRAY){
            if (value instanceof  NumpyArray){
                ndVars.put(name, (NumpyArray)value);
            }
            else if (value instanceof  INDArray) {
                ndVars.put(name, new NumpyArray((INDArray) value));
            }
            else{
                throw new RuntimeException("Unsupported type: " + value.getClass().toString());
            }
        }
        else if (type == Type.LIST){
            if (value instanceof List){
                value = ((List) value).toArray();
            }
            listVariables.put(name, (Object[]) value);
        }
        else if(type == Type.DICT) {
            dictVariables.put(name,(Map<?,?>) value);
        }
        else if (type == Type.FILE){
            fileVariables.put(name, (String)value);
        }
        else{
            strVariables.put(name, (String)value);
        }
    }

    /**
     * Do a general object lookup.
     * The look up will happen relative to the {@link Type}
     * of variable is described in the
     * @param name the name of the variable to get
     * @return teh value for the variable with the given name
     */
    public Object getValue(String name) {
        Type type = vars.get(name);
        Map map = maps.get(type);
        return map.get(name);
    }


    /**
     * Returns a boolean variable with the given name.
     * @param name the variable name to get the value for
     * @return the retrieved boolean value
     */
    public boolean getBooleanValue(String name) {
        return boolVariables.get(name);
    }

    /**
     *
     * @param name the variable name
     * @return the dictionary value
     */
    public Map<?,?> getDictValue(String name) {
        return dictVariables.get(name);
    }

    /**
     /**
     *
     * @param name the variable name
     * @return the string value
     */
    public String getStrValue(String name){
        return strVariables.get(name);
    }

    /**
     *
     * @param name the variable name
     * @return the long value
     */
    public Long getIntValue(String name){
        return intVariables.get(name);
    }

    /**
     *
     * @param name the variable name
     * @return the float value
     */
    public Double getFloatValue(String name){
        return floatVariables.get(name);
    }

    /**
     *
     * @param name the variable name
     * @return the numpy array value
     */
    public NumpyArray getNDArrayValue(String name){
        return ndVars.get(name);
    }

    /**
     *
     * @param name the variable name
     * @return the list value as an object array
     */
    public Object[] getListValue(String name){
        return listVariables.get(name);
    }

    /**
     *
     * @param name the variable name
     * @return the value of the given file name
     */
    public String getFileValue(String name){
        return fileVariables.get(name);
    }

    /**
     * Returns the type for the given variable name
     * @param name the name of the variable to get the type for
     * @return the type for the given variable
     */
    public Type getType(String name){
        return vars.get(name);
    }

    /**
     * Get all the variables present as a string array
     * @return the variable names for this variable sset
     */
    public String[] getVariables() {
        String[] strArr = new String[vars.size()];
        return vars.keySet().toArray(strArr);
    }


    /**
     * This variables set as its json representation (an array of json objects)
     * @return the json array output
     */
    public JSONArray toJSON(){
        JSONArray arr = new JSONArray();
        for (String varName: getVariables()){
            JSONObject var = new JSONObject();
            var.put("name", varName);
            String varType = getType(varName).toString();
            var.put("type", varType);
            arr.add(var);
        }
        return arr;
    }

    /**
     * Create a schema from a map.
     * This is an empty PythonVariables
     * that just contains names and types with no values
     * @param inputTypes the input types to convert
     * @return the schema from the given map
     */
    public static PythonVariables schemaFromMap(Map<String,String> inputTypes) {
        PythonVariables ret = new PythonVariables();
        for(Map.Entry<String,String> entry : inputTypes.entrySet()) {
            ret.add(entry.getKey(), Type.valueOf(entry.getValue()));
        }

        return ret;
    }

    /**
     * Get the python variable state relative to the
     * input json array
     * @param jsonArray the input json array
     * @return the python variables based on the input json array
     */
    public static PythonVariables fromJSON(JSONArray jsonArray){
        PythonVariables pyvars = new PythonVariables();
        for (int i = 0; i < jsonArray.size(); i++) {
            JSONObject input = (JSONObject) jsonArray.get(i);
            String varName = (String)input.get("name");
            String varType = (String)input.get("type");
            if (varType.equals("BOOL")){
                pyvars.addBool(varName);
            }
            else if (varType.equals("INT")){
                pyvars.addInt(varName);
            }
            else if (varType.equals("FlOAT")){
                pyvars.addFloat(varName);
            }
            else if (varType.equals("STR")){
                pyvars.addStr(varName);
            }
            else if (varType.equals("LIST")){
                pyvars.addList(varName);
            }
            else if (varType.equals("FILE")){
                pyvars.addFile(varName);
            }
            else if (varType.equals("NDARRAY")){
                pyvars.addNDArray(varName);
            }
            else if(varType.equals("DICT")) {
                pyvars.addDict(varName);
            }
        }

        return pyvars;
    }


}