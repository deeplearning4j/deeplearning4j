/*
 *  ******************************************************************************
 *  *
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
import org.bytedeco.javacpp.Pointer;

import java.util.*;

import static org.bytedeco.cpython.global.python.*;

public class PythonObject {

    static {
        new PythonExecutioner();
    }

    private boolean owned = true;
    private PyObject nativePythonObject;


    public PythonObject(PyObject nativePythonObject, boolean owned) {
        PythonGIL.assertThreadSafe();
        this.nativePythonObject = nativePythonObject;
        this.owned = owned;
        if (owned && nativePythonObject != null) {
            PythonGC.register(this);
        }
    }

    public PythonObject(PyObject nativePythonObject) {
        PythonGIL.assertThreadSafe();
        this.nativePythonObject = nativePythonObject;
        if (nativePythonObject != null) {
            PythonGC.register(this);
        }

    }

    public PyObject getNativePythonObject() {
        return nativePythonObject;
    }

    public String toString() {
        return PythonTypes.STR.toJava(this);

    }

    public boolean isNone() {
        if (nativePythonObject == null || Pointer.isNull(nativePythonObject)) {
            return true;
        }
        try (PythonGC _ = PythonGC.pause()) {
            PythonObject type = Python.type(this);
            boolean ret = Python.type(this).toString().equals("<class 'NoneType'>") && toString().equals("None");
            Py_DecRef(type.nativePythonObject);
            return ret;
        }
    }

    public void del() {
        PythonGIL.assertThreadSafe();
        if (owned && nativePythonObject != null && !PythonGC.isWatching()) {
            Py_DecRef(nativePythonObject);
            nativePythonObject = null;
        }
    }

    public PythonObject callWithArgs(PythonObject args) {
        return callWithArgsAndKwargs(args, null);
    }

    public PythonObject callWithKwargs(PythonObject kwargs) {
        if (!Python.callable(this)) {
            throw new PythonException("Object is not callable: " + toString());
        }
        PyObject tuple = PyTuple_New(0);
        PyObject dict = kwargs.nativePythonObject;
        if (PyObject_IsInstance(dict, new PyObject(PyDict_Type())) != 1) {
            throw new PythonException("Expected kwargs to be dict. Received: " + kwargs.toString());
        }
        PythonObject ret = new PythonObject(PyObject_Call(nativePythonObject, tuple, dict));
        Py_DecRef(tuple);
        return ret;
    }

    public PythonObject callWithArgsAndKwargs(PythonObject args, PythonObject kwargs) {
        PythonGIL.assertThreadSafe();
        PyObject tuple = null;
        boolean ownsTuple = false;
        try {
            if (!Python.callable(this)) {
                throw new PythonException("Object is not callable: " + toString());
            }

            if (PyObject_IsInstance(args.nativePythonObject, new PyObject(PyTuple_Type())) == 1) {
                tuple = args.nativePythonObject;
            } else if (PyObject_IsInstance(args.nativePythonObject, new PyObject(PyList_Type())) == 1) {
                tuple = PyList_AsTuple(args.nativePythonObject);
                ownsTuple = true;
            } else {
                throw new PythonException("Expected args to be tuple or list. Received: " + args.toString());
            }
            if (kwargs != null && PyObject_IsInstance(kwargs.nativePythonObject, new PyObject(PyDict_Type())) != 1) {
                throw new PythonException("Expected kwargs to be dict. Received: " + kwargs.toString());
            }
            return new PythonObject(PyObject_Call(nativePythonObject, tuple, kwargs == null ? null : kwargs.nativePythonObject));
        } finally {
            if (ownsTuple) Py_DecRef(tuple);
        }

    }


    public PythonObject call(Object... args) {
        return callWithArgsAndKwargs(Arrays.asList(args), null);
    }

    public PythonObject callWithArgs(List args) {
        return call(args, null);
    }

    public PythonObject callWithKwargs(Map kwargs) {
        return call(null, kwargs);
    }

    public PythonObject callWithArgsAndKwargs(List args, Map kwargs) {
        PythonGIL.assertThreadSafe();
        try (PythonGC _ = PythonGC.watch()) {
            if (!Python.callable(this)) {
                throw new PythonException("Object is not callable: " + toString());
            }
            PythonObject pyArgs;
            PythonObject pyKwargs;

            if (args == null || args.isEmpty()) {
                pyArgs = new PythonObject(PyTuple_New(0));
            } else {
                PythonObject argsList = PythonTypes.convert(args);
                pyArgs = new PythonObject(PyList_AsTuple(argsList.getNativePythonObject()));
            }
            if (kwargs == null) {
                pyKwargs = null;
            } else {
                pyKwargs = PythonTypes.convert(kwargs);
            }

            PythonObject ret = new PythonObject(
                    PyObject_Call(
                            nativePythonObject,
                            pyArgs.nativePythonObject,
                            pyKwargs == null ? null : pyKwargs.nativePythonObject
                    )
            );

            PythonGC.keep(ret);

            return ret;
        }

    }


    public PythonObject attr(String attrName) {
        PythonGIL.assertThreadSafe();
        return new PythonObject(PyObject_GetAttrString(nativePythonObject, attrName));
    }


    public PythonObject(Object javaObject) {
        PythonGIL.assertThreadSafe();
        if (javaObject instanceof PythonObject) {
            owned = false;
            nativePythonObject = ((PythonObject) javaObject).nativePythonObject;
        } else {
            try (PythonGC _ = PythonGC.pause()) {
                nativePythonObject = PythonTypes.convert(javaObject).getNativePythonObject();
            }
            PythonGC.register(this);
        }

    }

    public int toInt() {
        return PythonTypes.INT.toJava(this).intValue();
    }

    public long toLong() {
        return PythonTypes.INT.toJava(this);
    }

    public float toFloat() {
        return PythonTypes.FLOAT.toJava(this).floatValue();
    }

    public double toDouble() {
        return PythonTypes.FLOAT.toJava(this);
    }

    public boolean toBoolean() {
        return PythonTypes.BOOL.toJava(this);

    }

    public List toList() {
        return PythonTypes.LIST.toJava(this);
    }

    public Map toMap() {
        return PythonTypes.DICT.toJava(this);
    }

    public PythonObject get(int key) {
        PythonGIL.assertThreadSafe();
        return new PythonObject(PyObject_GetItem(nativePythonObject, PyLong_FromLong(key)));
    }

    public PythonObject get(String key) {
        PythonGIL.assertThreadSafe();
        return new PythonObject(PyObject_GetItem(nativePythonObject, PyUnicode_FromString(key)));
    }

    public PythonObject get(PythonObject key) {
        PythonGIL.assertThreadSafe();
        return new PythonObject(PyObject_GetItem(nativePythonObject, key.nativePythonObject));
    }

    public void set(PythonObject key, PythonObject value) {
        PythonGIL.assertThreadSafe();
        PyObject_SetItem(nativePythonObject, key.nativePythonObject, value.nativePythonObject);
    }


    public PythonObject abs(){
        return new PythonObject(PyNumber_Absolute(nativePythonObject));
    }
    public PythonObject add(PythonObject pythonObject){
        return new PythonObject(PyNumber_Add(nativePythonObject, pythonObject.nativePythonObject));
    }
    public PythonObject sub(PythonObject pythonObject){
        return new PythonObject(PyNumber_Subtract(nativePythonObject, pythonObject.nativePythonObject));
    }
    public PythonObject mod(PythonObject pythonObject){
        return new PythonObject(PyNumber_Divmod(nativePythonObject, pythonObject.nativePythonObject));
    }
    public PythonObject mul(PythonObject pythonObject){
        return new PythonObject(PyNumber_Multiply(nativePythonObject, pythonObject.nativePythonObject));
    }
    public PythonObject trueDiv(PythonObject pythonObject){
        return new PythonObject(PyNumber_TrueDivide(nativePythonObject, pythonObject.nativePythonObject));
    }
    public PythonObject floorDiv(PythonObject pythonObject){
        return new PythonObject(PyNumber_FloorDivide(nativePythonObject, pythonObject.nativePythonObject));
    }
    public PythonObject matMul(PythonObject pythonObject){
        return new PythonObject(PyNumber_MatrixMultiply(nativePythonObject, pythonObject.nativePythonObject));
    }

    public void addi(PythonObject pythonObject){
        PyNumber_InPlaceAdd(nativePythonObject, pythonObject.nativePythonObject);
    }
    public void subi(PythonObject pythonObject){
        PyNumber_InPlaceSubtract(nativePythonObject, pythonObject.nativePythonObject);
    }
    public void muli(PythonObject pythonObject){
        PyNumber_InPlaceMultiply(nativePythonObject, pythonObject.nativePythonObject);
    }
    public void trueDivi(PythonObject pythonObject){
        PyNumber_InPlaceTrueDivide(nativePythonObject, pythonObject.nativePythonObject);
    }
    public void floorDivi(PythonObject pythonObject){
        PyNumber_InPlaceFloorDivide(nativePythonObject, pythonObject.nativePythonObject);
    }
    public void matMuli(PythonObject pythonObject){
        PyNumber_InPlaceMatrixMultiply(nativePythonObject, pythonObject.nativePythonObject);
    }
}
