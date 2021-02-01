/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */


package org.nd4j.python4j;

import org.bytedeco.cpython.PyObject;

import java.util.Collections;
import java.util.List;

import static org.bytedeco.cpython.global.python.*;


public class Python {

    static {
        new PythonExecutioner();
    }

    /**
     * Imports a python module, similar to python import statement.
     *
     * @param moduleName name of the module to be imported
     * @return reference to the module object
     */
    public static PythonObject importModule(String moduleName) {
        PythonGIL.assertThreadSafe();
        PythonObject module = new PythonObject(PyImport_ImportModule(moduleName));
        if (module.isNone()) {
            throw new PythonException("Error importing module: " + moduleName);
        }
        return module;
    }

    /**
     * Gets a builtins attribute
     *
     * @param attrName Attribute name
     * @return
     */
    public static PythonObject attr(String attrName) {
        PythonGIL.assertThreadSafe();
        PyObject builtins = PyImport_ImportModule("builtins");
        try {
            return new PythonObject(PyObject_GetAttrString(builtins, attrName));
        } finally {
            Py_DecRef(builtins);
        }
    }


    /**
     * Gets the size of a PythonObject. similar to len() in python.
     *
     * @param pythonObject
     * @return
     */
    public static PythonObject len(PythonObject pythonObject) {
        PythonGIL.assertThreadSafe();
        long n = PyObject_Size(pythonObject.getNativePythonObject());
        if (n < 0) {
            throw new PythonException("Object has no length: " + pythonObject);
        }
        return PythonTypes.INT.toPython(n);
    }

    /**
     * Gets the string representation of an object.
     *
     * @param pythonObject
     * @return
     */
    public static PythonObject str(PythonObject pythonObject) {
        PythonGIL.assertThreadSafe();
        try {
            return PythonTypes.STR.toPython(pythonObject.toString());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }

    /**
     * Returns an empty string
     *
     * @return
     */
    public static PythonObject str() {
        PythonGIL.assertThreadSafe();
        try {
            return PythonTypes.STR.toPython("");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Returns the str type object
     * @return
     */
    public static PythonObject strType() {
        return attr("str");
    }

    /**
     * Returns a floating point number from a number or a string.
     * @param pythonObject
     * @return
     */
    public static PythonObject float_(PythonObject pythonObject) {
        return PythonTypes.FLOAT.toPython(PythonTypes.FLOAT.toJava(pythonObject));
    }

    /**
     * Reutrns 0.
     * @return
     */
    public static PythonObject float_() {
        try {
            return PythonTypes.FLOAT.toPython(0d);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    /**
     * Returns the float type object
     * @return
     */
    public static PythonObject floatType() {
        return attr("float");
    }


    /**
     * Converts a value to a Boolean value i.e., True or False, using the standard truth testing procedure.
     * @param pythonObject
     * @return
     */
    public static PythonObject bool(PythonObject pythonObject) {
        return PythonTypes.BOOL.toPython(PythonTypes.BOOL.toJava(pythonObject));

    }

    /**
     * Returns False.
     * @return
     */
    public static PythonObject bool() {
        return PythonTypes.BOOL.toPython(false);

    }

    /**
     * Returns the bool type object
     * @return
     */
    public static PythonObject boolType() {
        return attr("bool");
    }

    /**
     * Returns an integer from a number or a string.
     * @param pythonObject
     * @return
     */
    public static PythonObject int_(PythonObject pythonObject) {
        return PythonTypes.INT.toPython(PythonTypes.INT.toJava(pythonObject));
    }

    /**
     * Returns 0
     * @return
     */
    public static PythonObject int_() {
        return PythonTypes.INT.toPython(0L);

    }

    /**
     * Returns the int type object
     * @return
     */
    public static PythonObject intType() {
        return attr("int");
    }

    /**
     *  Takes sequence types and converts them to lists.
     * @param pythonObject
     * @return
     */
    public static PythonObject list(PythonObject pythonObject) {
        PythonGIL.assertThreadSafe();
        try (PythonGC _ = PythonGC.watch()) {
            PythonObject listF = attr("list");
            PythonObject ret = listF.call(pythonObject);
            if (ret.isNone()) {
                throw new PythonException("Object is not iterable: " + pythonObject.toString());
            }
            return ret;
        }
    }

    /**
     * Returns empty list.
     * @return
     */
    public static PythonObject list() {
        return PythonTypes.LIST.toPython(Collections.emptyList());
    }

    /**
     * Returns list type object.
     * @return
     */
    public static PythonObject listType() {
        return attr("list");
    }

    /**
     *  Creates a dictionary.
     * @param pythonObject
     * @return
     */
    public static PythonObject dict(PythonObject pythonObject) {
        PythonObject dictF = attr("dict");
        PythonObject ret = dictF.call(pythonObject);
        if (ret.isNone()) {
            throw new PythonException("Cannot build dict from object: " + pythonObject.toString());
        }
        dictF.del();
        return ret;
    }

    /**
     * Returns empty dict
     * @return
     */
    public static PythonObject dict() {
        return PythonTypes.DICT.toPython(Collections.emptyMap());
    }

    /**
     * Returns dict type object.
     * @return
     */
    public static PythonObject dictType() {
        return attr("dict");
    }

    /**
     * Creates a set.
     * @param pythonObject
     * @return
     */
    public static PythonObject set(PythonObject pythonObject) {
        PythonObject setF = attr("set");
        PythonObject ret = setF.call(pythonObject);
        if (ret.isNone()) {
            throw new PythonException("Cannot build set from object: " + pythonObject.toString());
        }
        setF.del();
        return ret;
    }

    /**
     * Returns empty set.
     * @return
     */
    public static PythonObject set() {
        PythonObject setF = attr("set");
        PythonObject ret;
        ret = setF.call();
        setF.del();
        return ret;
    }

    /**
     * Returns empty set.
     * @return
     */
    public static PythonObject setType() {
        return attr("set");
    }

    /**
     * Creates a bytearray.
     * @param pythonObject
     * @return
     */
    public static PythonObject bytearray(PythonObject pythonObject) {
        PythonObject baF = attr("bytearray");
        PythonObject ret = baF.call(pythonObject);
        if (ret.isNone()) {
            throw new PythonException("Cannot build bytearray from object: " + pythonObject.toString());
        }
        baF.del();
        return ret;
    }

    /**
     * Returns empty bytearray.
     * @return
     */
    public static PythonObject bytearray() {
        PythonObject baF = attr("bytearray");
        PythonObject ret;
        ret = baF.call();
        baF.del();
        return ret;
    }

    /**
     * Returns bytearray type object
     * @return
     */
    public static PythonObject bytearrayType() {
        return attr("bytearray");
    }

    /**
     * Creates a memoryview.
     * @param pythonObject
     * @return
     */
    public static PythonObject memoryview(PythonObject pythonObject) {
        PythonObject mvF = attr("memoryview");
        PythonObject ret = mvF.call(pythonObject);
        if (ret.isNone()) {
            throw new PythonException("Cannot build memoryview from object: " + pythonObject.toString());
        }
        mvF.del();
        return ret;
    }

    /**
     * Returns memoryview type object.
     * @return
     */
    public static PythonObject memoryviewType() {
        return attr("memoryview");
    }

    /**
     * Creates a byte string.
     * @param pythonObject
     * @return
     */
    public static PythonObject bytes(PythonObject pythonObject) {
        PythonObject bytesF = attr("bytes");
        PythonObject ret = bytesF.call(pythonObject);
        if (ret.isNone()) {
            throw new PythonException("Cannot build bytes from object: " + pythonObject.toString());
        }
        bytesF.del();
        return ret;
    }

    /**
     * Returns empty byte string.
     * @return
     */
    public static PythonObject bytes() {
        PythonObject bytesF = attr("bytes");
        PythonObject ret;
        ret = bytesF.call();
        bytesF.del();
        return ret;
    }

    /**
     * Returns bytes type object
     * @return
     */
    public static PythonObject bytesType() {
        return attr("bytes");
    }

    /**
     * Creates a tuple.
     * @param pythonObject
     * @return
     */
    public static PythonObject tuple(PythonObject pythonObject) {
        PythonObject tupleF = attr("tupleF");
        PythonObject ret = tupleF.call(pythonObject);
        if (ret.isNone()) {
            throw new PythonException("Cannot build tuple from object: " + pythonObject.toString());
        }
        tupleF.del();
        return ret;
    }

    /**
     * Returns empty tuple.
     * @return
     */
    public static PythonObject tuple() {
        PythonObject tupleF = attr("tuple");
        PythonObject ret;
        ret = tupleF.call();
        tupleF.del();
        return ret;
    }

    /**
     * Returns tuple type object
     * @return
     */
    public static PythonObject tupleType() {
        return attr("tuple");
    }

    /**
     * Creates an Exception
     * @param pythonObject
     * @return
     */
    public static PythonObject Exception(PythonObject pythonObject) {
        PythonObject excF = attr("Exception");
        PythonObject ret = excF.call(pythonObject);
        excF.del();
        return ret;
    }

    /**
     * Creates an Exception
     * @return
     */
    public static PythonObject Exception() {
        PythonObject excF = attr("Exception");
        PythonObject ret;
        ret = excF.call();
        excF.del();
        return ret;
    }

    /**
     * Returns Exception type object
     * @return
     */
    public static PythonObject ExceptionType() {
        return attr("Exception");
    }


    /**
     * Returns the globals dictionary.
     * @return
     */
    public static PythonObject globals() {
        PythonGIL.assertThreadSafe();
        PyObject main = PyImport_ImportModule("__main__");
        PyObject globals = PyModule_GetDict(main);
        Py_DecRef(main);
        return new PythonObject(globals, false);
    }

    /**
     * Returns the type of an object.
     * @param pythonObject
     * @return
     */
    public static PythonObject type(PythonObject pythonObject) {
        PythonObject typeF = attr("type");
        PythonObject ret = typeF.call(pythonObject);
        typeF.del();
        return ret;
    }

    /**
     * Returns True if the specified object is of the specified type, otherwise False.
     * @param obj
     * @param type
     * @return
     */
    public static boolean isinstance(PythonObject obj, PythonObject... type) {
        PythonGIL.assertThreadSafe();
        PyObject argsTuple = PyTuple_New(type.length);
        try {
            for (int i = 0; i < type.length; i++) {
                PythonObject x = type[i];
                Py_IncRef(x.getNativePythonObject());
                PyTuple_SetItem(argsTuple, i, x.getNativePythonObject());
            }
            return PyObject_IsInstance(obj.getNativePythonObject(), argsTuple) != 0;
        } finally {
            Py_DecRef(argsTuple);
        }

    }

    /**
     * Evaluates the specified expression.
     * @param expression
     * @return
     */
    public static PythonObject eval(String expression) {

        PythonGIL.assertThreadSafe();
        PyObject compiledCode = Py_CompileString(expression, "", Py_eval_input);
        PyObject main = PyImport_ImportModule("__main__");
        PyObject globals = PyModule_GetDict(main);
        PyObject locals = PyDict_New();
        try {
            return new PythonObject(PyEval_EvalCode(compiledCode, globals, locals));
        } finally {
            Py_DecRef(main);
            Py_DecRef(locals);
            Py_DecRef(compiledCode);
        }

    }

    /**
     * Returns the builtins module
     * @return
     */
    public static PythonObject builtins() {
        return importModule("builtins");

    }

    /**
     * Returns None.
     * @return
     */
    public static PythonObject None() {
        return eval("None");
    }

    /**
     * Returns True.
     * @return
     */
    public static PythonObject True() {
        return eval("True");
    }

    /**
     * Returns False.
     * @return
     */
    public static PythonObject False() {
        return eval("False");
    }

    /**
     * Returns True if the object passed is callable callable, otherwise False.
     * @param pythonObject
     * @return
     */
    public static boolean callable(PythonObject pythonObject) {
        PythonGIL.assertThreadSafe();
        return PyCallable_Check(pythonObject.getNativePythonObject()) == 1;
    }


    public static void setContext(String context){
        PythonContextManager.setContext(context);
    }

    public static String getCurrentContext() {
        return PythonContextManager.getCurrentContext();
    }

    public static void deleteContext(String context){
        PythonContextManager.deleteContext(context);
    }
    public static void resetContext() {
        PythonContextManager.reset();
    }

    /**
     * Executes a string of code.
     * @param code
     * @throws PythonException
     */
    public static void exec(String code) throws PythonException {
        PythonExecutioner.exec(code);
    }

    /**
     * Executes a string of code.
     * @param code
     * @param inputs
     * @param outputs
     */
    public static void exec(String code, List<PythonVariable> inputs, List<PythonVariable> outputs){
        PythonExecutioner.exec(code, inputs, outputs);
    }


}
