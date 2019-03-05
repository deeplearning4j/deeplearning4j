package org.datavec.python;

import java.io.File;
import java.io.FileInputStream;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import lombok.extern.slf4j.Slf4j;
import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;
import org.bytedeco.javacpp.*;
import static org.bytedeco.javacpp.python.*;

import org.nd4j.linalg.api.buffer.DataType;

@Slf4j
public class PythonExecutioner {
    private String name;
    private Pointer namePtr;
    private boolean restricted = false;
    private PyObject module;
    private PyObject globals;
    private JSONParser parser = new JSONParser();
    private Map<String, PyThreadState> interpreters = new HashMap<String, PyThreadState>();
    private String currentInterpreter =  null;
    private static PythonExecutioner pythonExecutioner;
    private boolean safeExecFlag = false;

    public static PythonExecutioner getInstance(){
        // do not use constructor
        if (pythonExecutioner == null){
            pythonExecutioner = new PythonExecutioner();
        }
        return pythonExecutioner;
    }
    public void setInterpreter(String name){
        if (name == null){
            if (currentInterpreter != null){
                PyThreadState_Swap(null);
                currentInterpreter = null;
                return;
            }
        }
        if (currentInterpreter != null && currentInterpreter.equals(name)){
            return;
        }
        else if (interpreters.containsKey(name)){
            PyThreadState threadState = interpreters.get(name);
            log.info("CPython: PyThreadState_Swap()");
            PyThreadState_Swap(threadState);
            init();
        }
        else{
            Py_Initialize();
            log.info("CPython: Py_NewInterpreter()");
            PyThreadState threadState = Py_NewInterpreter();
            interpreters.put(name, threadState);
            log.info("CPython: PyThreadState_Swap()");
            PyThreadState_Swap(threadState);
            init();
        }
        currentInterpreter = name;
    }

    public void deleteInterpreter(String name){
        if (name != null && !interpreters.containsKey(name)){
            return;
        }
        String temp = currentInterpreter;
        setInterpreter(name);
        log.info("CPython: Py_EndInterpreter()");
        Py_EndInterpreter(interpreters.get(name));
        interpreters.remove(name);
        if (temp == name){
            setInterpreter(null);
        }
        else{
            setInterpreter(temp);
        }
    }

    public PyObject getGlobals(){
        return globals;

    }

    public void setRestricted(boolean restricted) {
        this.restricted = restricted;
    }

    public boolean getRestricted(){
        return restricted;
    }

    public PythonExecutioner(String name){
        this.name = name;
        init();
    }


    public  PythonExecutioner(){
        this.name = PythonExecutioner.class.getSimpleName();
        init();
    }

    public void init(){
        log.info("CPython: Py_DecodeLocale()");
        namePtr = Py_DecodeLocale(name, null);
        log.info("CPython: Py_SetProgramName()");

        log.info("CPython: Py_Initialize()");
        Py_Initialize();
        log.info("CPython: PyImport_AddModule()");
        module = PyImport_AddModule("__main__");
        log.info("CPython: PyModule_GetDict()");
        globals = PyModule_GetDict(module);
    }

    public void free(){
        log.info("CPython: Py_FinalizeEx()");
        if (Py_FinalizeEx() < 0) {
            throw new RuntimeException("Python execution failed.");
        }
        log.info("CPython: PyMem_RawFree()");
        PyMem_RawFree(namePtr);
    }


    private String jArrayToPyString(Object[] array){
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

    private String escapeStr(String str){
        str = str.replace("\\", "\\\\");
        str = str.replace("\"\"\"", "\\\"\\\"\\\"");
        return str;
    }
    private String inputCode(PythonVariables pyInputs)throws Exception{
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
                String shapeStr = "(";
                for (long d: npArr.getShape()){
                    shapeStr += String.valueOf(d) + ",";
                }
                shapeStr += ")";
                String code;
                String ctype;
                if (npArr.getDType() == DataType.FLOAT){

                    ctype = "ctypes.c_float";
                }
                else if (npArr.getDType() == DataType.DOUBLE){
                    ctype = "ctypes.c_double";
                }
                else if (npArr.getDType() == DataType.SHORT){
                    ctype = "ctypes.c_int16";
                }
                else if (npArr.getDType() == DataType.INT){
                    ctype = "ctypes.c_int32";
                }
                else if (npArr.getDType() == DataType.LONG){
                    ctype = "ctypes.c_int64";
                }
                else{
                    throw new Exception("Unsupported data type: " + npArr.getDType().toString() + ".");
                }

                code = "__arr_converter(" + String.valueOf(npArr.getAddress()) + "," + shapeStr + "," + ctype + ")";
                code = varName + "=" + code + "\n";
                inputCode += code;
                inputCode += "loc['" + varName + "']=" + varName + "\n";
            }

        }
        return inputCode;
    }

    private void _readOutputs(PythonVariables pyOutputs){
        if (pyOutputs == null){
            return;
        }
        exec(getOutputCheckCode(pyOutputs));
        String errorMessage = evalSTRING("__error_message");
        if (errorMessage.length() > 0){
            throw new RuntimeException(errorMessage);
        }
        try{

            for (String varName: pyOutputs.getVariables()){
                PythonVariables.Type type = pyOutputs.getType(varName);
                if (type == PythonVariables.Type.STR){
                    pyOutputs.setValue(varName, evalSTRING(varName));
                }
                else if(type == PythonVariables.Type.FLOAT){
                    pyOutputs.setValue(varName, evalFLOAT(varName));
                }
                else if(type == PythonVariables.Type.INT){
                    pyOutputs.setValue(varName, evalINTEGER(varName));
                }
                else if (type == PythonVariables.Type.LIST){
                    Object varVal[] = evalLIST(varName);
                    pyOutputs.setValue(varName, varVal);
                }
                else{
                    pyOutputs.setValue(varName, evalNDARRAY(varName));
                }
            }
        }
        catch (Exception e){
            log.error(e.toString());
        }

    }



    public void exec(String code){
        log.info("CPython: PyRun_SimpleStringFlag()");
        log.info(code);
        PyRun_SimpleStringFlags(code, null);
        log.info("Exec done");
    }

    public void exec(String code, PythonVariables pyInputs, PythonVariables pyOutputs) throws Exception{
        String inputCode = inputCode(pyInputs);
        if (code.charAt(code.length() - 1) != '\n'){
            code += '\n';
        }
        if(restricted){
            code = RestrictedPython.getSafeCode(code);
        }
        exec(inputCode + code);
        _readOutputs(pyOutputs);
    }

    private void setupTransform(PythonTransform transform){
        String name = transform.getName();
        if (!interpreters.containsKey(name)){
            setInterpreter(name);
            String setupCode = transform.getSetupCode();
            if (setupCode != null) {
                exec(setupCode);
            }
        }
        else{
            setInterpreter(name);
        }
    }
    public PythonVariables exec(PythonTransform transform) throws Exception{
        setupTransform(transform);
        if (transform.getInputs() != null && transform.getInputs().getVariables().length > 0){
            throw new Exception("Required inputs not provided.");
        }
        exec(transform.getExecCode(), null, transform.getOutputs());
        return transform.getOutputs();
    }

    public PythonVariables exec(PythonTransform transform, PythonVariables inputs)throws Exception{
        setupTransform(transform);
        exec(transform.getExecCode(), inputs, transform.getOutputs());
        return transform.getOutputs();
    }



    // safe exec
    public PythonVariables safeExec(PythonTransform transform) throws Exception{
        setupTransform(transform);
        if (transform.getInputs() != null && transform.getInputs().getVariables().length > 0){
            throw new Exception("Required inputs not provided.");
        }
        safeExecFlag = true;
        exec(transform.getExecCode(), null, transform.getOutputs());
        safeExecFlag = false;
        //deleteInterpreter(transform.getName());
        return transform.getOutputs();
    }

    public PythonVariables safeExec(PythonTransform transform, PythonVariables inputs)throws Exception{
        setupTransform(transform);
        safeExecFlag = true;
        exec(transform.getExecCode(), inputs, transform.getOutputs());
        safeExecFlag = false;
        //deleteInterpreter(transform.getName());
        return transform.getOutputs();
    }

    public PythonVariables batchedExec(PythonTransform transform, PythonVariables inputs) throws Exception{
        String[] varNames = inputs.getVariables();
        Map<Integer, PythonVariables> inputBatches = new HashMap<>();
        for (String varNameAndIdx: varNames){
            String[]  split = varNameAndIdx.split(Pattern.quote("["));
            String varName = split[0];
            int batchId = Integer.parseInt(split[1].split(Pattern.quote("]"))[0]);
            if (!inputBatches.containsKey(batchId)){
                inputBatches.put(batchId, new PythonVariables());
            }
            inputBatches.get(batchId).add(varName, inputs.getType(varNameAndIdx), inputs.getValue(varNameAndIdx));
        }
        PythonVariables outputs = new PythonVariables();
        for (int batch=0; batch < inputBatches.size(); batch++){
            PythonVariables batchOutput = exec(transform, inputBatches.get(batch));
            for (String varName: batchOutput.getVariables()){
                outputs.add(varName + "[" + batch + "]", batchOutput.getType(varName), batchOutput.getValue(varName));
            }
        }
        return outputs;
    }


    public PythonVariables batchedSafeExec(PythonTransform transform, PythonVariables inputs) throws Exception{
        String[] varNames = inputs.getVariables();
        Map<Integer, PythonVariables> inputBatches = new HashMap<>();
        for (String varNameAndIdx: varNames){
            String[]  split = varNameAndIdx.split(Pattern.quote("["));
            String varName = split[0];
            int batchId = Integer.parseInt(split[1].split(Pattern.quote("]"))[0]);
            if (!inputBatches.containsKey(batchId)){
                inputBatches.put(batchId, new PythonVariables());
            }
            inputBatches.get(batchId).add(varName, inputs.getType(varNameAndIdx), inputs.getValue(varNameAndIdx));
        }
        PythonVariables outputs = new PythonVariables();
        for (int batch=0; batch < inputBatches.size(); batch++){
            safeExecFlag = true;
            PythonVariables batchOutput = exec(transform, inputBatches.get(batch));
            safeExecFlag = false;
            for (String varName: batchOutput.getVariables()){
                outputs.add(varName + "[" + batch + "]", batchOutput.getType(varName), batchOutput.getValue(varName));
            }
        }
        deleteInterpreter(transform.getName());
        return outputs;
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

    public String evalSTRING(String varName){
        PyObject xObj = PyDict_GetItemString(globals, varName);
        PyObject bytes = PyUnicode_AsEncodedString(xObj, "UTF-8", "strict");
        BytePointer bp = PyBytes_AsString(bytes);
        String ret = bp.getString();
        Py_DecRef(xObj);
        Py_DecRef(bytes);
        return ret;
    }

    public long evalINTEGER(String varName){
        PyObject xObj = PyDict_GetItemString(globals, varName);
        long ret = PyLong_AsLongLong(xObj);
        return ret;
    }

    public double evalFLOAT(String varName){
        PyObject xObj = PyDict_GetItemString(globals, varName);
        double ret = PyFloat_AsDouble(xObj);
        return ret;
    }

    public Object[] evalLIST(String varName) throws Exception{
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

    public NumpyArray evalNDARRAY(String varName) throws Exception{
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
        NumpyArray ret = new NumpyArray(address, shape, strides, dtype, safeExecFlag);
        Py_DecRef(arrayInterface);
        Py_DecRef(data);
        Py_DecRef(zero);
        Py_DecRef(addressObj);
        Py_DecRef(shapeObj);
        Py_DecRef(stridesObj);


       return ret;
    }

    private String getOutputCheckCode(PythonVariables pyOutputs){
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
}
