
/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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


import org.bytedeco.cpython.PyObject;

import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.PyArray_EnsureArray;

/**
 * Swift like python wrapper for Java
 *
 * @author Fariz Rahman
 */

public class Python {

    /**
     * Imports a python module, similar to python import statement.
     * @param moduleName name of the module to be imported
     * @return reference to the module object
     * @throws PythonException
     */
    public static PythonObject importModule(String moduleName) throws PythonException{
        PythonObject module = new PythonObject(PyImport_ImportModule(moduleName));
        if (module.isNone()) {
            throw new PythonException("Error importing module: " + moduleName);
        }
        return module;
    }

    public static PythonObject attr(String attrName) {
        return builtins().attr(attrName);
    }

    public static PythonObject len(PythonObject pythonObject) {
        return attr("len").call(pythonObject);
    }

    public static PythonObject str(PythonObject pythonObject) {
        return attr("str").call(pythonObject);
    }

    public static PythonObject str() {
        return attr("str").call();
    }

    public static PythonObject strType() {
        return attr("str");
    }

    public static PythonObject float_(PythonObject pythonObject) {
        return attr("float").call(pythonObject);
    }

    public static PythonObject float_() {
        return attr("float").call();
    }

    public static PythonObject floatType() {
        return attr("float");
    }

    public static PythonObject bool(PythonObject pythonObject) {
        return attr("bool").call(pythonObject);
    }

    public static PythonObject bool() {
        return attr("bool").call();
    }

    public static PythonObject boolType() {
        return attr("bool");
    }

    public static PythonObject int_(PythonObject pythonObject) {
        return attr("int").call(pythonObject);
    }

    public static PythonObject int_() {
        return attr("int").call();
    }

    public static PythonObject intType() {
        return attr("int");
    }

    public static PythonObject list(PythonObject pythonObject) {
        return attr("list").call(pythonObject);
    }

    public static PythonObject list() {
        return attr("list").call();
    }

    public static PythonObject listType() {
        return attr("list");
    }

    public static PythonObject dict(PythonObject pythonObject) {
        return attr("dict").call(pythonObject);
    }

    public static PythonObject dict() {
        return attr("dict").call();
    }

    public static PythonObject dictType() {
        return attr("dict");
    }

    public static PythonObject set(PythonObject pythonObject) {
        return attr("set").call(pythonObject);
    }

    public static PythonObject set() {
        return attr("set").call();
    }

    public static PythonObject bytearray(PythonObject pythonObject) {
        return attr("bytearray").call(pythonObject);
    }

    public static PythonObject bytearray() {
        return attr("bytearray").call();
    }

    public static PythonObject bytearrayType() {
        return attr("bytearray");
    }

    public static PythonObject memoryview(PythonObject pythonObject) {
        return attr("memoryview").call(pythonObject);
    }

    public static PythonObject memoryviewType() {
        return attr("memoryview");
    }

    public static PythonObject bytes(PythonObject pythonObject) {
        return attr("bytes").call(pythonObject);
    }

    public static PythonObject bytes() {
        return attr("bytes").call();
    }

    public static PythonObject bytesType() {
        return attr("bytes");
    }

    public static PythonObject tuple(PythonObject pythonObject) {
        return attr("tuple").call(pythonObject);
    }

    public static PythonObject tuple() {
        return attr("tuple").call();
    }


    public static PythonObject Exception(PythonObject pythonObject) {
        return attr("Exception").call(pythonObject);
    }

    public static PythonObject Exception() {
        return attr("Exception").call();
    }

    public static PythonObject ExceptionType() {
        return attr("Exception");
    }


    public static PythonObject tupleType() {
        return attr("tuple");
    }
    public static PythonObject globals() {
        return new PythonObject(PyModule_GetDict(PyImport_ImportModule("__main__")));
    }

    public static PythonObject type(PythonObject obj) {
        return attr("type").call(obj);
    }

    public static boolean isinstance(PythonObject obj, PythonObject... type) {
        return PyObject_IsInstance(obj.getNativePythonObject(),
                PyList_AsTuple(new PythonObject(type).getNativePythonObject())) != 0;
    }

    public static PythonObject eval(String code) {
        PyObject compiledCode = Py_CompileString(code, "", Py_eval_input);
        PyObject globals = globals().getNativePythonObject();
        PyObject locals = Python.dict().getNativePythonObject();
        return new PythonObject(PyEval_EvalCode(compiledCode, globals, locals));
    }


    public static PythonObject builtins(){
        try{
            return importModule("builtins");
        }catch (PythonException pe){
            throw new IllegalStateException("Unable to import builtins: " + pe); // this should never happen
        }

    }

    public static PythonObject None() {
        return dict().attr("get").call(0);
    }

    public static PythonObject True() {
        return boolType().call(1);
    }

    public static PythonObject False() {
        return boolType().call(0);

    }

    public static PythonObject ndarray(PythonObject pythonObject){
        return new PythonObject(PyArray_EnsureArray(pythonObject.getNativePythonObject()));
    }

    public static boolean callable(PythonObject pythonObject) {
        return PyCallable_Check(pythonObject.getNativePythonObject()) == 1;
    }


    public static void setContext(String context) throws PythonException{
        PythonContextManager.setContext(context);
    }

    public static String getCurrentContext(){
        return PythonContextManager.getCurrentContext();
    }

    public static void deleteContext(String context) throws PythonException{
        PythonContextManager.deleteContext(context);
    }

    public static void deleteNonMainContexts(){
        PythonContextManager.deleteNonMainContexts();
    }

    public static void setMainContext(){PythonContextManager.setMainContext();}

    public static void exec(String code)throws PythonException{
        PythonExecutioner.exec(code);
    }
    public static void exec(String code, PythonVariables inputs, PythonVariables outputs) throws PythonException{
        PythonExecutioner.exec(code, inputs, outputs);
    }

    public static PythonGIL lock(){
        return PythonGIL.lock();
    }


}
