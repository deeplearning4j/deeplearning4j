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
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.json.JSONObject;
import org.json.JSONArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.Serializable;
import java.nio.ByteBuffer;
import java.util.*;



/**
 * Holds python variable names, types and values.
 * Also handles mapping from java types to python types.
 *
 * @author Fariz Rahman
 */

@lombok.Data
public class PythonVariables implements java.io.Serializable {
    

    private java.util.Map<String, String> strVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, Long> intVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, Double> floatVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, Boolean> boolVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, INDArray> ndVars = new java.util.LinkedHashMap<>();
    private java.util.Map<String, List> listVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, BytePointer> bytesVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, java.util.Map<?, ?>> dictVariables = new java.util.LinkedHashMap<>();
    private java.util.Map<String, PythonType.TypeName> vars = new java.util.LinkedHashMap<>();
    private java.util.Map<PythonType.TypeName, java.util.Map> maps = new java.util.LinkedHashMap<>();


    /**
     * Returns a copy of the variable
     * schema in this array without the values
     *
     * @return an empty variables clone
     * with no values
     */
    public PythonVariables copySchema() {
        PythonVariables ret = new PythonVariables();
        for (String varName : getVariables()) {
            PythonType type = getType(varName);
            ret.add(varName, type);
        }
        return ret;
    }

    /**
     *
     */
    public PythonVariables() {
        maps.put(PythonType.TypeName.BOOL, boolVariables);
        maps.put(PythonType.TypeName.STR, strVariables);
        maps.put(PythonType.TypeName.INT, intVariables);
        maps.put(PythonType.TypeName.FLOAT, floatVariables);
        maps.put(PythonType.TypeName.NDARRAY, ndVars);
        maps.put(PythonType.TypeName.LIST, listVariables);
        maps.put(PythonType.TypeName.DICT, dictVariables);
        maps.put(PythonType.TypeName.BYTES, bytesVariables);

    }


    /**
     * @return true if there are no variables.
     */
    public boolean isEmpty() {
        return getVariables().length < 1;
    }


    /**
     * @param name Name of the variable
     * @param type Type of the variable
     */
    public void add(String name, PythonType type) {
        switch (type.getName()) {
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
            case DICT:
                addDict(name);
                break;
            case BYTES:
                addBytes(name);
                break;
        }
    }

    /**
     * @param name  name of the variable
     * @param type  type of the variable
     * @param value value of the variable (must be instance of expected type)
     */
    public void add(String name, PythonType type, Object value) throws PythonException {
        add(name, type);
        setValue(name, value);
    }


    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     *
     * @param name the field to add
     */
    public void addDict(String name) {
        vars.put(name, PythonType.TypeName.DICT);
        dictVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     *
     * @param name the field to add
     */
    public void addBool(String name) {
        vars.put(name, PythonType.TypeName.BOOL);
        boolVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     *
     * @param name the field to add
     */
    public void addStr(String name) {
        vars.put(name, PythonType.TypeName.STR);
        strVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     *
     * @param name the field to add
     */
    public void addInt(String name) {
        vars.put(name, PythonType.TypeName.INT);
        intVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     *
     * @param name the field to add
     */
    public void addFloat(String name) {
        vars.put(name, PythonType.TypeName.FLOAT);
        floatVariables.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     *
     * @param name the field to add
     */
    public void addNDArray(String name) {
        vars.put(name, PythonType.TypeName.NDARRAY);
        ndVars.put(name, null);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     *
     * @param name the field to add
     */
    public void addList(String name) {
        vars.put(name, PythonType.TypeName.LIST);
        listVariables.put(name, null);
    }

    /**
     * Add a boolean variable to
     * the set of variables
     *
     * @param name  the field to add
     * @param value the value to add
     */
    public void addBool(String name, boolean value) {
        vars.put(name, PythonType.TypeName.BOOL);
        boolVariables.put(name, value);
    }

    /**
     * Add a string variable to
     * the set of variables
     *
     * @param name  the field to add
     * @param value the value to add
     */
    public void addStr(String name, String value) {
        vars.put(name, PythonType.TypeName.STR);
        strVariables.put(name, value);
    }

    /**
     * Add an int variable to
     * the set of variables
     *
     * @param name  the field to add
     * @param value the value to add
     */
    public void addInt(String name, int value) {
        vars.put(name, PythonType.TypeName.INT);
        intVariables.put(name, (long) value);
    }

    /**
     * Add a long variable to
     * the set of variables
     *
     * @param name  the field to add
     * @param value the value to add
     */
    public void addInt(String name, long value) {
        vars.put(name, PythonType.TypeName.INT);
        intVariables.put(name, value);
    }

    /**
     * Add a double variable to
     * the set of variables
     *
     * @param name  the field to add
     * @param value the value to add
     */
    public void addFloat(String name, double value) {
        vars.put(name, PythonType.TypeName.FLOAT);
        floatVariables.put(name, value);
    }

    /**
     * Add a float variable to
     * the set of variables
     *
     * @param name  the field to add
     * @param value the value to add
     */
    public void addFloat(String name, float value) {
        vars.put(name, PythonType.TypeName.FLOAT);
        floatVariables.put(name, (double) value);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     *
     * @param name  the field to add
     * @param value the value to add
     */
    public void addNDArray(String name, NumpyArray value) {
        vars.put(name, PythonType.TypeName.NDARRAY);
        ndVars.put(name, value.getNd4jArray());
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     *
     * @param name  the field to add
     * @param value the value to add
     */
    public void addNDArray(String name, INDArray value) {
        vars.put(name, PythonType.TypeName.NDARRAY);
        ndVars.put(name, value);
    }

    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     *
     * @param name  the field to add
     * @param value the value to add
     */
    public void addList(String name, Object[] value) {
        vars.put(name, PythonType.TypeName.LIST);
        listVariables.put(name, Arrays.asList(value));
    }
    
    /**
     * Add a null variable to
     * the set of variables
     * to describe the type but no value
     *
     * @param name  the field to add
     * @param value the value to add
     */
    public void addDict(String name, java.util.Map value) {
        vars.put(name, PythonType.TypeName.DICT);
        dictVariables.put(name, value);
    }


    public void addBytes(String name){
        vars.put(name, PythonType.TypeName.BYTES);
        bytesVariables.put(name, null);
    }

    public void addBytes(String name, BytePointer value){
        vars.put(name, PythonType.TypeName.BYTES);
        bytesVariables.put(name, value);
    }

//    public void addBytes(String name, ByteBuffer value){
//        Pointer ptr = NativeOpsHolder.getInstance().getDeviceNativeOps().pointerForAddress((value.address());
//        BytePointer bp = new BytePointer(ptr);
//        addBytes(name, bp);
//    }
    /**
     * @param name  name of the variable
     * @param value new value for the variable
     */
    public void setValue(String name, Object value) throws PythonException {
        PythonType.TypeName type = vars.get(name);
        maps.get(type).put(name, PythonType.valueOf(type).convert(value));
    }

    /**
     * Do a general object lookup.
     * The look up will happen relative to the {@link PythonType}
     * of variable is described in the
     *
     * @param name the name of the variable to get
     * @return teh value for the variable with the given name
     */
    public Object getValue(String name) {
        PythonType.TypeName type = vars.get(name);
        java.util.Map map = maps.get(type);
        return map.get(name);
    }


    /**
     * Returns a boolean variable with the given name.
     *
     * @param name the variable name to get the value for
     * @return the retrieved boolean value
     */
    public boolean getBooleanValue(String name) {
        return boolVariables.get(name);
    }

    /**
     * @param name the variable name
     * @return the dictionary value
     */
    public java.util.Map<?, ?> getDictValue(String name) {
        return dictVariables.get(name);
    }

    /**
     * /**
     *
     * @param name the variable name
     * @return the string value
     */
    public String getStrValue(String name) {
        return strVariables.get(name);
    }

    /**
     * @param name the variable name
     * @return the long value
     */
    public Long getIntValue(String name) {
        return intVariables.get(name);
    }

    /**
     * @param name the variable name
     * @return the float value
     */
    public Double getFloatValue(String name) {
        return floatVariables.get(name);
    }

    /**
     * @param name the variable name
     * @return the numpy array value
     */
    public INDArray getNDArrayValue(String name) {
        return ndVars.get(name);
    }

    /**
     * @param name the variable name
     * @return the list value as an object array
     */
    public List getListValue(String name) {
        return listVariables.get(name);
    }

    /**
     * @param name the variable name
     * @return the bytes value as a BytePointer
     */
    public BytePointer getBytesValue(String name){return bytesVariables.get(name);}
    /**
     * Returns the type for the given variable name
     *
     * @param name the name of the variable to get the type for
     * @return the type for the given variable
     */
    public PythonType getType(String name){
        try{
            return PythonType.valueOf(vars.get(name));  // will never fail
        }catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }

    /**
     * Get all the variables present as a string array
     *
     * @return the variable names for this variable sset
     */
    public String[] getVariables() {
        String[] strArr = new String[vars.size()];
        return vars.keySet().toArray(strArr);
    }


    /**
     * This variables set as its json representation (an array of json objects)
     *
     * @return the json array output
     */
    public org.json.JSONArray toJSON() {
        org.json.JSONArray arr = new org.json.JSONArray();
        for (String varName : getVariables()) {
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
     *
     * @param inputTypes the input types to convert
     * @return the schema from the given map
     */
    public static PythonVariables schemaFromMap(java.util.Map<String, String> inputTypes) throws Exception{
        PythonVariables ret = new PythonVariables();
        for (java.util.Map.Entry<String, String> entry : inputTypes.entrySet()) {
            ret.add(entry.getKey(), PythonType.valueOf(entry.getValue()));
        }

        return ret;
    }

    /**
     * Get the python variable state relative to the
     * input json array
     *
     * @param jsonArray the input json array
     * @return the python variables based on the input json array
     */
    public static PythonVariables fromJSON(org.json.JSONArray jsonArray) {
        PythonVariables pyvars = new PythonVariables();
        for (int i = 0; i < jsonArray.length(); i++) {
            org.json.JSONObject input = (org.json.JSONObject) jsonArray.get(i);
            String varName = (String) input.get("name");
            String varType = (String) input.get("type");
            pyvars.maps.get(PythonType.TypeName.valueOf(varType)).put(varName, null);
        }

        return pyvars;
    }


}