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


import java.io.File;
import java.io.FileInputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

import lombok.extern.slf4j.Slf4j;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.bytedeco.javacpp.*;
import org.bytedeco.cpython.*;
import static org.bytedeco.cpython.global.python.*;
import org.nd4j.linalg.api.buffer.DataType;

/**
 *  Python executioner
 *
 *  @author Fariz Rahman
 */
@Slf4j
public class PythonExecutioner {
    private static PyObject module;
    private static PyObject globals;
    private static JSONParser parser = new JSONParser();
    private static Map<Long, Integer> gilStates = new HashMap<>();

    static {
        init();
    }

    public static void init(){
        log.info("CPython: Py_InitializeEx()");
        Py_InitializeEx(1);
        log.info("CPython: PyEval_InitThreads()");
        PyEval_InitThreads();
        log.info("CPython: PyImport_AddModule()");
        module = PyImport_AddModule("__main__");
        log.info("CPython: PyModule_GetDict()");
        globals = PyModule_GetDict(module);
        log.info("CPython: PyThreadState_Get()");
        PyEval_SaveThread();
    }

    public static void free(){
        Py_Finalize();
    }

    private static String inputCode(PythonVariables pyInputs)throws Exception{
        String inputCode = "loc={};";
        if (pyInputs == null){
            return inputCode;
        }
        Map<String, String> strInputs = pyInputs.getStrVariables();
        Map<String, Long> intInputs = pyInputs.getIntVariables();
        Map<String, Double> floatInputs = pyInputs.getFloatVariables();
        Map<String, NumpyArray> ndInputs = pyInputs.getNDArrayVariables();
        Map<String, Object[]> listInputs = pyInputs.getListVariables();
        Map<String, String> fileInputs = pyInputs.getFileVariables();

        String[] VarNames;


        VarNames = strInputs.keySet().toArray(new String[strInputs.size()]);
        for(Object varName: VarNames){
            String varValue = strInputs.get(varName);
            inputCode += varName + " = \"\"\"" + escapeStr(varValue) + "\"\"\"\n";
            inputCode += "loc['" + varName + "']=" + varName + "\n";
        }

        VarNames = intInputs.keySet().toArray(new String[intInputs.size()]);
        for(String varName: VarNames){
            Long varValue = intInputs.get(varName);
            inputCode += varName + " = " + varValue.toString() + "\n";
            inputCode += "loc['" + varName + "']=" + varName + "\n";
        }

        VarNames = floatInputs.keySet().toArray(new String[floatInputs.size()]);
        for(String varName: VarNames){
            Double varValue = floatInputs.get(varName);
            inputCode += varName + " = " + varValue.toString() + "\n";
            inputCode += "loc['" + varName + "']=" + varName + "\n";
        }

        VarNames = listInputs.keySet().toArray(new String[listInputs.size()]);
        for (String varName: VarNames){
            Object[] varValue = listInputs.get(varName);
            String listStr = jArrayToPyString(varValue);
            inputCode += varName + " = " + listStr + "\n";
            inputCode += "loc['" + varName + "']=" + varName + "\n";
        }

        VarNames = fileInputs.keySet().toArray(new String[fileInputs.size()]);
        for(Object varName: VarNames){
            String varValue = fileInputs.get(varName);
            inputCode += varName + " = \"\"\"" + escapeStr(varValue) + "\"\"\"\n";
            inputCode += "loc['" + varName + "']=" + varName + "\n";
        }

        if (ndInputs.size()> 0){
            inputCode += "import ctypes; import numpy as np;";
            VarNames = ndInputs.keySet().toArray(new String[ndInputs.size()]);

            String converter = "__arr_converter = lambda addr, shape, type: np.ctypeslib.as_array(ctypes.cast(addr, ctypes.POINTER(type)), shape);";
            inputCode += converter;
            for(String varName: VarNames){
                NumpyArray npArr = ndInputs.get(varName);
                npArr = npArr.copy();
                String shapeStr = "(";
                for (long d: npArr.getShape()){
                    shapeStr += String.valueOf(d) + ",";
                }
                shapeStr += ")";
                String code;
                String ctype;
                if (npArr.getDtype() == DataType.FLOAT){

                    ctype = "ctypes.c_float";
                }
                else if (npArr.getDtype() == DataType.DOUBLE){
                    ctype = "ctypes.c_double";
                }
                else if (npArr.getDtype() == DataType.SHORT){
                    ctype = "ctypes.c_int16";
                }
                else if (npArr.getDtype() == DataType.INT){
                    ctype = "ctypes.c_int32";
                }
                else if (npArr.getDtype() == DataType.LONG){
                    ctype = "ctypes.c_int64";
                }
                else{
                    throw new Exception("Unsupported data type: " + npArr.getDtype().toString() + ".");
                }

                code = "__arr_converter(" + String.valueOf(npArr.getAddress()) + "," + shapeStr + "," + ctype + ")";
                code = varName + "=" + code + "\n";
                inputCode += code;
                inputCode += "loc['" + varName + "']=" + varName + "\n";
            }

        }
        return inputCode;
    }


    private static void _readOutputs(PythonVariables pyOutputs){
        String json = read(getTempFile());
        File f = new File(getTempFile());
        f.delete();
        JSONParser p = new JSONParser();
        try{
            JSONObject jobj = (JSONObject) p.parse(json);
            for (String varName: pyOutputs.getVariables()){
                PythonVariables.Type type = pyOutputs.getType(varName);
                if (type == PythonVariables.Type.NDARRAY){
                    JSONObject varValue = (JSONObject)jobj.get(varName);
                    long address = (Long)varValue.get("address");
                    JSONArray shapeJson = (JSONArray)varValue.get("shape");
                    JSONArray stridesJson = (JSONArray)varValue.get("strides");
                    long[] shape = jsonArrayToLongArray(shapeJson);
                    long[] strides = jsonArrayToLongArray(stridesJson);
                    String dtypeName = (String)varValue.get("dtype");
                    DataType dtype;
                    if (dtypeName.equals("float64")){
                        dtype = DataType.DOUBLE;
                    }
                    else if (dtypeName.equals("float32")){
                        dtype = DataType.FLOAT;
                    }
                    else if (dtypeName.equals("int16")){
                        dtype = DataType.SHORT;
                    }
                    else if (dtypeName.equals("int32")){
                        dtype = DataType.INT;
                    }
                    else if (dtypeName.equals("int64")){
                        dtype = DataType.LONG;
                    }
                    else{
                        throw new Exception("Unsupported array type " + dtypeName + ".");
                    }
                    pyOutputs.setValue(varName, new NumpyArray(address, shape, strides, dtype, true));


                }
                else if (type == PythonVariables.Type.LIST){
                    JSONArray varValue = (JSONArray)jobj.get(varName);
                    pyOutputs.setValue(varName, varValue.toArray());
                }
                else{
                    pyOutputs.setValue(varName, jobj.get(varName));
                }
            }
        }
        catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    private static void acquireGIL(){
        log.info("---_enterSubInterpreter()---");
        if (PyGILState_Check() != 1){
            gilStates.put(Thread.currentThread().getId(), PyGILState_Ensure());
            log.info("GIL ensured");
        }
    }

    private static void releaseGIL(){
        if (PyGILState_Check() == 1){
            log.info("Releasing gil...");
            PyGILState_Release(gilStates.get(Thread.currentThread().getId()));
            log.info("Gil released.");
        }

    }

    /**
     * Executes python code. Also manages python thread state.
     * @param code
     */
    public static void exec(String code){

        code = getFunctionalCode("__f_" + Thread.currentThread().getId(), code);
        acquireGIL();
        log.info("CPython: PyRun_SimpleStringFlag()");
        log.info(code);
        int result = PyRun_SimpleStringFlags(code, null);
        if (result != 0){
            PyErr_Print();
            throw new RuntimeException("exec failed");
        }
        log.info("Exec done");
        releaseGIL();
    }

    public static void exec(String code, PythonVariables pyOutputs){
        exec(code + '\n' + outputCode(pyOutputs));
        _readOutputs(pyOutputs);
    }

    public static void exec(String code, PythonVariables pyInputs, PythonVariables pyOutputs) throws Exception{
        String inputCode = inputCode(pyInputs);
        exec(inputCode + code, pyOutputs);
    }


    public static PythonVariables exec(PythonTransform transform) throws Exception{
        if (transform.getInputs() != null && transform.getInputs().getVariables().length > 0){
            throw new Exception("Required inputs not provided.");
        }
        exec(transform.getCode(), null, transform.getOutputs());
        return transform.getOutputs();
    }

    public static PythonVariables exec(PythonTransform transform, PythonVariables inputs)throws Exception{
        exec(transform.getCode(), inputs, transform.getOutputs());
        return transform.getOutputs();
    }


    public static String evalSTRING(String varName){
        log.info("CPython: PyImport_AddModule()");
        module = PyImport_AddModule("__main__");
        log.info("CPython: PyModule_GetDict()");
        globals = PyModule_GetDict(module);
        PyObject xObj = PyDict_GetItemString(globals, varName);
        PyObject bytes = PyUnicode_AsEncodedString(xObj, "UTF-8", "strict");
        BytePointer bp = PyBytes_AsString(bytes);
        String ret = bp.getString();
        Py_DecRef(xObj);
        Py_DecRef(bytes);
        return ret;
    }

    public static long evalINTEGER(String varName){
        log.info("CPython: PyImport_AddModule()");
        module = PyImport_AddModule("__main__");
        log.info("CPython: PyModule_GetDict()");
        globals = PyModule_GetDict(module);
        PyObject xObj = PyDict_GetItemString(globals, varName);
        long ret = PyLong_AsLongLong(xObj);
        return ret;
    }

    public static double evalFLOAT(String varName){
        log.info("CPython: PyImport_AddModule()");
        module = PyImport_AddModule("__main__");
        log.info("CPython: PyModule_GetDict()");
        globals = PyModule_GetDict(module);
        PyObject xObj = PyDict_GetItemString(globals, varName);
        double ret = PyFloat_AsDouble(xObj);
        return ret;
    }

    public static Object[] evalLIST(String varName) throws Exception{
        log.info("CPython: PyImport_AddModule()");
        module = PyImport_AddModule("__main__");
        log.info("CPython: PyModule_GetDict()");
        globals = PyModule_GetDict(module);
        PyObject xObj = PyDict_GetItemString(globals, varName);
        PyObject strObj = PyObject_Str(xObj);
        PyObject bytes = PyUnicode_AsEncodedString(strObj, "UTF-8", "strict");
        BytePointer bp = PyBytes_AsString(bytes);
        String listStr = bp.getString();
        Py_DecRef(xObj);
        Py_DecRef(bytes);
        JSONArray jsonArray = (JSONArray)parser.parse(listStr.replace("\'", "\""));
        return jsonArray.toArray();
    }

    public static NumpyArray evalNDARRAY(String varName) throws Exception{
        log.info("CPython: PyImport_AddModule()");
        module = PyImport_AddModule("__main__");
        log.info("CPython: PyModule_GetDict()");
        globals = PyModule_GetDict(module);
        PyObject xObj = PyDict_GetItemString(globals, varName);
        PyObject arrayInterface = PyObject_GetAttrString(xObj, "__array_interface__");
        PyObject data = PyDict_GetItemString(arrayInterface, "data");
        PyObject zero = PyLong_FromLong(0);
        PyObject addressObj = PyObject_GetItem(data, zero);
        long address = PyLong_AsLongLong(addressObj);
        PyObject shapeObj = PyObject_GetAttrString(xObj, "shape");
        int ndim = (int)PyObject_Size(shapeObj);
        PyObject iObj;
        long shape[] = new long[ndim];
        for (int i=0; i<ndim; i++){
            iObj = PyLong_FromLong(i);
            PyObject sizeObj = PyObject_GetItem(shapeObj, iObj);
            long size = PyLong_AsLongLong(sizeObj);
            shape[i] = size;
            Py_DecRef(iObj);
        }

        PyObject stridesObj = PyObject_GetAttrString(xObj, "strides");
        long strides[] = new long[ndim];
        for (int i=0; i<ndim; i++){
            iObj = PyLong_FromLong(i);
            PyObject strideObj = PyObject_GetItem(stridesObj, iObj);
            long stride = PyLong_AsLongLong(strideObj);
            strides[i] = stride;
            Py_DecRef(iObj);
        }

        PyObject dtypeObj = PyObject_GetAttrString(xObj, "dtype");
        PyObject dtypeNameObj = PyObject_GetAttrString(dtypeObj, "name");
        PyObject bytes = PyUnicode_AsEncodedString(dtypeNameObj, "UTF-8", "strict");
        BytePointer bp = PyBytes_AsString(bytes);
        String dtypeName = bp.getString();
        DataType dtype;
        if (dtypeName.equals("float64")){
            dtype = DataType.DOUBLE;
        }
        else if (dtypeName.equals("float32")){
            dtype = DataType.FLOAT;
        }
        else if (dtypeName.equals("int16")){
            dtype = DataType.SHORT;
        }
        else if (dtypeName.equals("int32")){
            dtype = DataType.INT;
        }
        else if (dtypeName.equals("int64")){
            dtype = DataType.LONG;
        }
        else{
            throw new Exception("Unsupported array type " + dtypeName + ".");
        }
        NumpyArray ret = new NumpyArray(address, shape, strides, dtype, true);
        Py_DecRef(arrayInterface);
        Py_DecRef(data);
        Py_DecRef(zero);
        Py_DecRef(addressObj);
        Py_DecRef(shapeObj);
        Py_DecRef(stridesObj);

        return ret;
    }

    private static String getOutputCheckCode(PythonVariables pyOutputs){
        // make sure all outputs exist and are of expected types
        // helps avoid JVM crashes (most of the time)
        String code= "__error_message=''\n";
        String checkVarExists = "if '%s' not in locals(): __error_message += '%s not found.'\n";
        String checkVarType = "if not isinstance(%s, %s): __error_message += '%s is not of required type.'\n";
        for (String varName: pyOutputs.getVariables()){
            PythonVariables.Type type = pyOutputs.getType(varName);
            code += String.format(checkVarExists, varName, varName);
            switch(type){
                case INT:
                    code += String.format(checkVarType, varName, "int", varName);
                    break;
                case STR:
                    code += String.format(checkVarType, varName, "str", varName);
                    break;
                case FLOAT:
                    code += String.format(checkVarType, varName, "float", varName);
                    break;
                case BOOL:
                    code += String.format(checkVarType, varName, "bool", varName);
                    break;
                case NDARRAY:
                    code += String.format(checkVarType, varName, "np.ndarray", varName);
                    break;
                case LIST:
                    code += String.format(checkVarType, varName, "list", varName);
                    break;
            }
        }
        return code;
    }

    private static  String outputCode(PythonVariables pyOutputs){

        if (pyOutputs == null){
            return "";
        }
        String outputCode = "import json\nwith open('" + getTempFile() + "', 'w') as ___fobj_:json.dump({";
        String[] VarNames = pyOutputs.getVariables();
        boolean ndarrayHelperAdded = false;
        for (String varName: VarNames){

            if (pyOutputs.getType(varName) == PythonVariables.Type.NDARRAY){
                if (! ndarrayHelperAdded){
                    ndarrayHelperAdded = true;
                    String helper = "serialize_ndarray_metadata=lambda x:{\"address\":x.__array_interface__['data'][0]" +
                            ",\"shape\":x.shape,\"strides\":x.strides,\"dtype\":str(x.dtype)}\n";
                    outputCode = helper + outputCode;
                }
                outputCode += "\"" + varName + "\"" + ":serialize_ndarray_metadata(" + varName + "),";

            }
            else {
                outputCode += "\"" + varName + "\"" + ":" + varName + ",";
            }
        }

        outputCode = outputCode.substring(0, outputCode.length() - 1);
        outputCode += "}, ___fobj_)\n";
        return outputCode;

    }
    private static String read(String path){
        try{
            File file = new File(path);
            FileInputStream fis = new FileInputStream(file);
            byte[] data = new byte[(int) file.length()];
            fis.read(data);
            fis.close();
            String str = new String(data, "UTF-8");
            return str;
        }
        catch (Exception e){
            return "";
        }

    }
    private static String jArrayToPyString(Object[] array){
        String str = "[";
        for (int i=0; i < array.length; i++){
            Object obj = array[i];
            if (obj instanceof Object[]){
                str += jArrayToPyString((Object[])obj);
            }
            else if (obj instanceof String){
                str += "\"" + obj + "\"";
            }
            else{
                str += obj.toString().replace("\"", "\\\"");
            }
            if (i < array.length - 1){
                str += ",";
            }

        }
        str += "]";
        return str;
    }

    private static String escapeStr(String str){
        str = str.replace("\\", "\\\\");
        str = str.replace("\"\"\"", "\\\"\\\"\\\"");
        return str;
    }

    private static String getFunctionalCode(String functionName, String code){
        String out = String.format("def %s():\n", functionName);
        for(String line: code.split(Pattern.quote("\n"))){
            out += "    " + line + "\n";
        }
        return out + "\n\n" + functionName + "()\n";
    }


    private static String getTempFile(){
        String ret =  "temp_" + Thread.currentThread().getId() + ".json";
        log.info(ret);
        return ret;
    }

    private static long[] jsonArrayToLongArray(JSONArray jsonArray){
        long[] longs = new long[jsonArray.size()];
        for (int i=0; i<longs.length; i++){
            longs[i] = (Long)jsonArray.get(i);
        }
        return longs;
    }
}
