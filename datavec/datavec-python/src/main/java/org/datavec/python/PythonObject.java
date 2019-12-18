package org.datavec.python;


import org.bytedeco.cpython.PyObject;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.cpython.global.python.PyObject_SetItem;

/**
 * Swift like python wrapper for J
 *
 * @author Fariz Rahman
 */

public class PythonObject {
    private PyObject nativePythonObject;
    public PythonObject(PyObject pyObject){
        nativePythonObject = pyObject;
    }
    public PythonObject(INDArray npArray){
        this(new NumpyArray(npArray));
    }
    public PythonObject(NumpyArray npArray){
        PyObject np = PyImport_AddModule("numpy");
        PyObject ctypes = PyImport_AddModule("ctypes");
        PyObject ctype;
        switch (npArray.getDtype()){
            case DOUBLE:
                ctype = PyObject_GetAttrString(ctypes, "c_double");
                break;
            case FLOAT:
                ctype = PyObject_GetAttrString(ctypes, "c_float");
                break;
            case LONG:
                ctype = PyObject_GetAttrString(ctypes, "c_int64");
                break;
            case INT:
                ctype = PyObject_GetAttrString(ctypes, "c_int32");
                break;
            case SHORT:
                ctype = PyObject_GetAttrString(ctypes, "c_int16");
                break;

            default:
                throw new RuntimeException("Unsupported dtype.");
        }

        PythonObject ptrType = new PythonObject(PyObject_GetAttrString(ctypes, "POINTER")).call(ctype);
        PythonObject ptr = new PythonObject(PyObject_GetAttrString(ctypes, "cast")).call(npArray.getAddress(), ptrType);
        nativePythonObject = new PythonObject(np).attr("ctypeslib").attr("as_array").call(
                ptr,
                new PythonObject(
                        PyList_AsTuple(
                                (new PythonObject(Arrays.asList(
                                        npArray.getShape()
                                )).nativePythonObject
                                )
                        )
                )).nativePythonObject;

    }

    /*---primitve constructors---*/
    public PyObject getNativePythonObject() {
        return nativePythonObject;
    }

    public PythonObject(String data){
        nativePythonObject = PyUnicode_FromString(data);
    }
    public PythonObject(int data){
        nativePythonObject = PyLong_FromLong((long)data);
    }
    public PythonObject(long data){
        nativePythonObject = PyLong_FromLong(data);
    }

    public PythonObject(double data){
        nativePythonObject = PyFloat_FromDouble(data);
    }

    public PythonObject(boolean data){
        nativePythonObject = PyBool_FromLong(data?1:0);
    }
    /*---collection constructors---*/

    public PythonObject(List data){
        PyObject pyList = PyList_New((long)data.size());
        for(int i=0; i < data.size(); i++){
            Object item = data.get(i);
            if (item instanceof PythonObject){
                PyList_SetItem(pyList, (long)i, ((PythonObject)item).nativePythonObject);
            }
            else if (item instanceof INDArray){
                PyList_SetItem(pyList, (long)i, new PythonObject((INDArray) item).nativePythonObject);
            }
            else if (item instanceof NumpyArray){
                PyList_SetItem(pyList, (long)i, new PythonObject((NumpyArray) item).nativePythonObject);
            }
            else if(item instanceof List){
                PyList_SetItem(pyList, (long)i, new PythonObject((List)item).nativePythonObject);
            }
            else if(item instanceof Map){
                PyList_SetItem(pyList, (long)i, new PythonObject((Map)item).nativePythonObject);
            }
            else if (item instanceof String){
                PyList_SetItem(pyList, (long)i, new PythonObject((String)item).nativePythonObject);
            }
            else if (item instanceof Double){
                PyList_SetItem(pyList, (long)i, new PythonObject((Double)item).nativePythonObject);
            }
            else if (item instanceof Float){
                PyList_SetItem(pyList, (long)i, new PythonObject((Float)item).nativePythonObject);
            }
            else if (item instanceof Long){
                PyList_SetItem(pyList, (long)i, new PythonObject((Long)item).nativePythonObject);
            }
            else if (item instanceof Integer){
                PyList_SetItem(pyList, (long)i, new PythonObject((Integer)item).nativePythonObject);
            }
            else if (item instanceof Boolean){
                PyList_SetItem(pyList, (long)i, new PythonObject((Boolean) item).nativePythonObject);
            }
            else{
                throw new RuntimeException("Unsupported item in list");
            }

        }
        nativePythonObject = pyList;
    }
    public PythonObject(Map data){
        PyObject pyDict = PyDict_New();
        for (Object k: data.keySet()){
            PythonObject pyKey;
            if (k instanceof  PythonObject){
                pyKey = (PythonObject)k;
            }
            else if (k instanceof String){
                pyKey = new PythonObject((String)k);
            }
            else if (k instanceof Double){
                pyKey = new PythonObject((Double)k);
            }
            else if (k instanceof Float){
                pyKey = new PythonObject((Float)k);
            }
            else if (k instanceof Long){
                pyKey = new PythonObject((Long)k);
            }
            else if (k instanceof Integer){
                pyKey = new PythonObject((Integer)k);
            }
            else if (k instanceof Boolean){
                pyKey = new PythonObject((Boolean)k);
            }
            else{
                throw new RuntimeException("Unsupported key in map");
            }
            Object v = data.get(k);
            PythonObject pyVal;
            if (v instanceof PythonObject){
                pyVal = (PythonObject)v;
            }
            else if (v instanceof INDArray){
                pyVal = new PythonObject((INDArray)v);
            }
            else if (v instanceof NumpyArray){
                pyVal = new PythonObject((NumpyArray) v);
            }
            else if (v instanceof Map){
                pyVal = new PythonObject((Map)v);
            }
            else if (v instanceof List){
                pyVal = new PythonObject((List)v);
            }
            else if (v instanceof String){
                pyVal = new PythonObject((String)v);
            }
            else if (v instanceof Double){
                pyVal = new PythonObject((Double)v);
            }
            else if (v instanceof Float){
                pyVal = new PythonObject((Float)v);
            }
            else if (v instanceof Long){
                pyVal = new PythonObject((Long)v);
            }
            else if (v instanceof Integer){
                pyVal = new PythonObject((Integer)v);
            }
            else if (v instanceof Boolean){
                pyVal = new PythonObject((Boolean)v);
            }
            else{
                throw new RuntimeException("Unsupported value in map");
            }

            PyDict_SetItem(pyDict, pyKey.nativePythonObject, pyVal.nativePythonObject);
        }
    }


    /*------*/

    public String toString(){
        PyObject repr = PyObject_Str(nativePythonObject);
        PyObject str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
        String jstr = PyBytes_AsString(str).getString();
        Py_DecRef(repr);
        Py_DecRef(str);
        return jstr;
    }

    public double toDouble(){
        return PyFloat_AsDouble(nativePythonObject);
    }
    public float toFloat(){
        return (float)PyFloat_AsDouble(nativePythonObject);
    }
    public int toInt(){
        return (int)PyLong_AsLong(nativePythonObject);
    }
    public long toLong(){
        return PyLong_AsLong(nativePythonObject);
    }
    public boolean toBoolean(){
        return toInt() != 0;
    }

    public NumpyArray toNumpy(){
        PyObject arrInterface = PyObject_GetAttrString(nativePythonObject, "__array_interface__");
        PyObject data = PyDict_GetItemString(arrInterface, "data");
        long address = PyLong_AsLong(PyTuple_GetItem(data, 0));
        String dtypeName = attr("dtype").attr("name").toString();
        PyObject shape = PyObject_GetAttrString(nativePythonObject, "shape");
        PyObject strides = PyObject_GetAttrString(nativePythonObject, "strides");
        int ndim = (int)PyObject_Size(shape);
        long[] jshape = new long[ndim];
        long[] jstrides = new long[ndim];
        for (int i=0; i<ndim;i++){
            jshape[i] = PyLong_AsLong(PyTuple_GetItem(shape, i));
            jstrides[i] = PyLong_AsLong(PyTuple_GetItem(strides, i));
        }
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
            throw new RuntimeException("Unsupported array type " + dtypeName + ".");
        }
        return new NumpyArray(address, jshape, jstrides, dtype);

    }
    public PythonObject attr(String attr){

        return new PythonObject(PyObject_GetAttrString(nativePythonObject, attr));
    }
    public PythonObject call(Object... args){
        PyObject tuple = PyList_AsTuple(new PythonObject(Arrays.asList(args)).nativePythonObject);
        return new PythonObject(PyObject_Call(nativePythonObject, tuple, null));
    }

    public PythonObject call(Map kwargs){
        PyObject dict = new PythonObject(kwargs).nativePythonObject;
        PyObject tuple = PyTuple_New(0);
        return new PythonObject(PyObject_Call(nativePythonObject, tuple, dict));
    }

    public PythonObject call(List args){
        PyObject tuple = PyList_AsTuple(new PythonObject(args).nativePythonObject);
        return new PythonObject(PyObject_Call(nativePythonObject, tuple, null));
    }
    public PythonObject call(List args, Map kwargs){
        PyObject tuple = PyList_AsTuple(new PythonObject(args).nativePythonObject);
        PyObject dict = new PythonObject(kwargs).nativePythonObject;
        return new PythonObject(PyObject_Call(nativePythonObject, tuple, dict));
    }
    private PythonObject get(PyObject key){
        return new PythonObject(
                PyObject_GetItem(nativePythonObject, key)
        );
    }
    public PythonObject get(PythonObject key){
        return get(key.nativePythonObject);
    }


    public PythonObject get(int key){
        return get(PyLong_FromLong((long)key));
    }

    public PythonObject get(long key){
        return new PythonObject(
                PyObject_GetItem(nativePythonObject, PyLong_FromLong(key))

        );
    }

    public PythonObject get(double key){
        return new PythonObject(
                PyObject_GetItem(nativePythonObject, PyFloat_FromDouble(key))

        );
    }

    public void set(PythonObject key, PythonObject value){
        PyObject_SetItem(nativePythonObject, key.nativePythonObject, value.nativePythonObject);
    }



}
