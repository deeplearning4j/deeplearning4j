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
import org.json.JSONObject;
import org.json.JSONArray;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.*;

/**
 * Holds python variable names, types and values.
 * Also handles mapping from java types to python types.
 *
 * @author Fariz Rahman
 */

@lombok.Data
public class PythonVariables implements java.io.Serializable {

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

    private java.util.Map<String, String> strVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, Long> intVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, Double> floatVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, Boolean> boolVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, NumpyArray> ndVars = new java.util.LinkedHashMap<>();
    private java.util.Map<String, Object[]> listVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, String> fileVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, java.util.Map<?,?>> dictVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, Type> vars = new java.util.LinkedHashMap<>();
    private java.util.Map<Type, java.util.Map> maps = new java.util.LinkedHashMap<>();


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
        maps.put(PythonVariables.Type.BOOL, boolVariables);
        maps.put(PythonVariables.Type.STR, strVariables);
        maps.put(PythonVariables.Type.INT, intVariables);
        maps.put(PythonVariables.Type.FLOAT, floatVariables);
        maps.put(PythonVariables.Type.NDARRAY, ndVars);
        maps.put(PythonVariables.Type.LIST, listVariables);
        maps.put(PythonVariables.Type.FILE, fileVariables);
        maps.put(PythonVariables.Type.DICT, dictVariables);

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
        vars.put(name, PythonVariables.Type.DICT);
        dictVariables.put(name,null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addBool(String name){
        vars.put(name, PythonVariables.Type.BOOL);
        boolVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addStr(String name){
        vars.put(name, PythonVariables.Type.STR);
        strVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addInt(String name){
        vars.put(name, PythonVariables.Type.INT);
        intVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addFloat(String name){
        vars.put(name, PythonVariables.Type.FLOAT);
        floatVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addNDArray(String name){
        vars.put(name, PythonVariables.Type.NDARRAY);
        ndVars.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addList(String name){
        vars.put(name, PythonVariables.Type.LIST);
        listVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     */
    public void addFile(String name){
        vars.put(name, PythonVariables.Type.FILE);
        fileVariables.put(name, null);
    }

    /**
     * Add a boolean variable to
     * the set of variables
     * @param name the field to add
     * @param value the value to add
     */
    public void addBool(String name, boolean value) {
        vars.put(name, PythonVariables.Type.BOOL);
        boolVariables.put(name, value);
    }

    /**
     * Add a string variable to
     * the set of variables
     * @param name the field to add
     * @param value the value to add
     */
    public void addStr(String name, String value) {
        vars.put(name, PythonVariables.Type.STR);
        strVariables.put(name, value);
    }

    /**
     * Add an int variable to
     * the set of variables
     * @param name the field to add
     * @param value the value to add
     */
    public void addInt(String name, int value) {
        vars.put(name, PythonVariables.Type.INT);
        intVariables.put(name, (long)value);
    }

    /**
     * Add a long variable to
     * the set of variables
     * @param name the field to add
     * @param value the value to add
     */
    public void addInt(String name, long value) {
        vars.put(name, PythonVariables.Type.INT);
        intVariables.put(name, value);
    }

    /**
     * Add a double variable to
     * the set of variables
     * @param name the field to add
     * @param value the value to add
     */
    public void addFloat(String name, double value) {
        vars.put(name, PythonVariables.Type.FLOAT);
        floatVariables.put(name, value);
    }

    /**
     * Add a float variable to
     * the set of variables
     * @param name the field to add
     * @param value the value to add
     */
    public void addFloat(String name, float value) {
        vars.put(name, PythonVariables.Type.FLOAT);
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
        vars.put(name, PythonVariables.Type.NDARRAY);
        ndVars.put(name, value);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     * @param value the value to add
     */
    public void addNDArray(String name, org.nd4j.linalg.api.ndarray.INDArray value) {
        vars.put(name, PythonVariables.Type.NDARRAY);
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
        vars.put(name, PythonVariables.Type.LIST);
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
        vars.put(name, PythonVariables.Type.FILE);
        fileVariables.put(name, value);
    }


    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     * @param name the field to add
     * @param value the value to add
     */
    public void addDict(String name, java.util.Map value) {
        vars.put(name, PythonVariables.Type.DICT);
        dictVariables.put(name, value);
    }
    /**
     *
     * @param name name of the variable
     * @param value new value for the variable
     */
    public void setValue(String name, Object value) {
        Type type = vars.get(name);
        if (type == PythonVariables.Type.BOOL){
            boolVariables.put(name, (Boolean)value);
        }
        else if (type == PythonVariables.Type.INT){
            Number number = (Number) value;
            intVariables.put(name, number.longValue());
        }
        else if (type == PythonVariables.Type.FLOAT){
            Number number = (Number) value;
            floatVariables.put(name, number.doubleValue());
        }
        else if (type == PythonVariables.Type.NDARRAY){
            if (value instanceof  NumpyArray){
                ndVars.put(name, (NumpyArray)value);
            }
            else if (value instanceof  org.nd4j.linalg.api.ndarray.INDArray) {
                ndVars.put(name, new NumpyArray((org.nd4j.linalg.api.ndarray.INDArray) value));
            }
            else{
                throw new RuntimeException("Unsupported type: " + value.getClass().toString());
            }
        }
        else if (type == PythonVariables.Type.LIST) {
            if (value instanceof java.util.List) {
                value = ((java.util.List) value).toArray();
                listVariables.put(name,  (Object[]) value);
            }
            else if(value instanceof org.json.JSONArray) {
                org.json.JSONArray jsonArray = (org.json.JSONArray) value;
                Object[] copyArr = new Object[jsonArray.length()];
                for(int i = 0; i < copyArr.length; i++) {
                    copyArr[i] = jsonArray.get(i);
                }
                listVariables.put(name,  copyArr);

            }
            else {
                listVariables.put(name,  (Object[]) value);
            }
        }
        else if(type == PythonVariables.Type.DICT) {
            dictVariables.put(name,(java.util.Map<?,?>) value);
        }
        else if (type == PythonVariables.Type.FILE){
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
        java.util.Map map = maps.get(type);
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
    public java.util.Map<?,?> getDictValue(String name) {
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
    public org.json.JSONArray toJSON(){
        org.json.JSONArray arr = new org.json.JSONArray();
        for (String varName: getVariables()){
            org.json.JSONObject var = new org.json.JSONObject();
            var.put("name", varName);
            String varType = getType(varName).toString();
            var.put("type", varType);
            arr.put(var);
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
    public static PythonVariables schemaFromMap(java.util.Map<String,String> inputTypes) {
        PythonVariables ret = new PythonVariables();
        for(java.util.Map.Entry<String,String> entry : inputTypes.entrySet()) {
            ret.add(entry.getKey(), PythonVariables.Type.valueOf(entry.getValue()));
        }

        return ret;
    }

    /**
     * Get the python variable state relative to the
     * input json array
     * @param jsonArray the input json array
     * @return the python variables based on the input json array
     */
    public static PythonVariables fromJSON(org.json.JSONArray jsonArray){
        PythonVariables pyvars = new PythonVariables();
        for (int i = 0; i < jsonArray.length(); i++) {
            org.json.JSONObject input = (org.json.JSONObject) jsonArray.get(i);
            String varName = (String)input.get("name");
            String varType = (String)input.get("type");
            if (varType.equals("BOOL")) {
                pyvars.addBool(varName);
            }
            else if (varType.equals("INT")) {
                pyvars.addInt(varName);
            }
            else if (varType.equals("FlOAT")){
                pyvars.addFloat(varName);
            }
            else if (varType.equals("STR")) {
                pyvars.addStr(varName);
            }
            else if (varType.equals("LIST")) {
                pyvars.addList(varName);
            }
            else if (varType.equals("FILE")){
                pyvars.addFile(varName);
            }
            else if (varType.equals("NDARRAY")) {
                pyvars.addNDArray(varName);
            }
            else if(varType.equals("DICT")) {
                pyvars.addDict(varName);
            }
        }

        return pyvars;
    }


}