package org.datavec.python;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import static org.bytedeco.cpython.global.python.PyImport_ImportModule;
import static org.bytedeco.cpython.global.python.PyObject_Size;

/**
 * Swift like python wrapper for J
 *
 * @author Fariz Rahman
 */

public class Python {
    public static PythonObject importModule(String moduleName){
        return new PythonObject(PyImport_ImportModule(moduleName));
    }
    public static long len(PythonObject pythonObject){
        return PyObject_Size(pythonObject.getNativePythonObject());
    }
    public static String str(PythonObject pythonObject){
        return pythonObject.toString();
    }
    public static double float_(PythonObject pythonObject){
        return pythonObject.toFloat();
    }
    public static  boolean bool(PythonObject pythonObject){
        return pythonObject.toBoolean();
    }
    public static long int_(PythonObject pythonObject){
        return pythonObject.toLong();
    }
    public static ArrayList list(PythonObject pythonObject){
        throw new RuntimeException("not implemented");
    }
    public static HashMap dict(PythonObject pythonObject){
        throw new RuntimeException("not implemented");
    }
    public static HashSet set(PythonObject pythonObject){
        throw new RuntimeException("not implemented");
    }
}
