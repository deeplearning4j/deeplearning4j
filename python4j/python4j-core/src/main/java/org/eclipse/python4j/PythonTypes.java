/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

package org.eclipse.python4j;


import org.bytedeco.cpython.PyObject;

import java.util.*;

import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.cpython.global.python.Py_DecRef;

public class PythonTypes {


    private static List<PythonType> getPrimitiveTypes() {
        return Arrays.<PythonType>asList(STR, INT, FLOAT, BOOL);
    }

    private static List<PythonType> getCollectionTypes() {
        return Arrays.<PythonType>asList(LIST, DICT);
    }

    private static List<PythonType> getExternalTypes() {
        //TODO service loader
        return new ArrayList<>();
    }

    public static List<PythonType> get() {
        List<PythonType> ret = new ArrayList<>();
        ret.addAll(getPrimitiveTypes());
        ret.addAll(getCollectionTypes());
        ret.addAll(getExternalTypes());
        return ret;
    }

    public static PythonType get(String name) {
        for (PythonType pt : get()) {
            if (pt.getName().equals(name)) {  // TODO use map instead?
                return pt;
            }
        }
        throw new PythonException("Unknown python type: " + name);
    }

    public static PythonType getPythonTypeForJavaObject(Object javaObject) {
        for (PythonType pt : get()) {
            if (pt.accepts(javaObject)) {
                return pt;
            }
        }
        throw new PythonException("Unable to find python type for java type: " + javaObject.getClass());
    }

    public static PythonType getPythonTypeForPythonObject(PythonObject pythonObject) {
        PyObject pyType = PyObject_Type(pythonObject.getNativePythonObject());
        try {
            String pyTypeStr = PythonTypes.STR.toJava(new PythonObject(pyType, false));

            for (PythonType pt : get()) {
                String pyTypeStr2 = "<class '" + pt.getName() + "'>";
                if (pyTypeStr.equals(pyTypeStr2)) {
                    return pt;
                }
            }
            throw new PythonException("Unable to find converter for python object of type " + pyTypeStr);
        } finally {
            Py_DecRef(pyType);
        }


    }

    public static PythonObject convert(Object javaObject) {
        PythonType pt = getPythonTypeForJavaObject(javaObject);
        return pt.toPython(pt.adapt(javaObject));
    }

    public static final PythonType<String> STR = new PythonType<String>("str", String.class) {

        @Override
        public String adapt(Object javaObject) {
            if (javaObject instanceof String) {
                return (String) javaObject;
            }
            throw new PythonException("Cannot cast object of type " + javaObject.getClass().getName() + " to String");
        }

        @Override
        public String toJava(PythonObject pythonObject) {
            PythonGIL.assertThreadSafe();
            PyObject repr = PyObject_Str(pythonObject.getNativePythonObject());
            PyObject str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
            String jstr = PyBytes_AsString(str).getString();
            Py_DecRef(repr);
            Py_DecRef(str);
            return jstr;
        }

        @Override
        public PythonObject toPython(String javaObject) {
            return new PythonObject(PyUnicode_FromString(javaObject));
        }
    };

    public static final PythonType<Long> INT = new PythonType<Long>("int", Long.class) {
        @Override
        public Long adapt(Object javaObject) {
            if (javaObject instanceof Number) {
                return ((Number) javaObject).longValue();
            }
            throw new PythonException("Cannot cast object of type " + javaObject.getClass().getName() + " to Long");
        }

        @Override
        public Long toJava(PythonObject pythonObject) {
            PythonGIL.assertThreadSafe();
            long val = PyLong_AsLong(pythonObject.getNativePythonObject());
            if (val == -1 && PyErr_Occurred() != null) {
                throw new PythonException("Could not convert value to int: " + pythonObject.toString());
            }
            return val;
        }

        @Override
        public boolean accepts(Object javaObject) {
            return (javaObject instanceof Integer) || (javaObject instanceof Long);
        }

        @Override
        public PythonObject toPython(Long javaObject) {
            return new PythonObject(PyLong_FromLong(javaObject));
        }
    };

    public static final PythonType<Double> FLOAT = new PythonType<Double>("float", Double.class) {

        @Override
        public Double adapt(Object javaObject) {
            if (javaObject instanceof Number) {
                return ((Number) javaObject).doubleValue();
            }
            throw new PythonException("Cannot cast object of type " + javaObject.getClass().getName() + " to Long");
        }

        @Override
        public Double toJava(PythonObject pythonObject) {
            PythonGIL.assertThreadSafe();
            double val = PyFloat_AsDouble(pythonObject.getNativePythonObject());
            if (val == -1 && PyErr_Occurred() != null) {
                throw new PythonException("Could not convert value to float: " + pythonObject.toString());
            }
            return val;
        }

        @Override
        public boolean accepts(Object javaObject) {
            return (javaObject instanceof Float) || (javaObject instanceof Double);
        }

        @Override
        public PythonObject toPython(Double javaObject) {
            return new PythonObject(PyFloat_FromDouble(javaObject));
        }
    };


    public static final PythonType<Boolean> BOOL = new PythonType<Boolean>("bool", Boolean.class) {

        @Override
        public Boolean adapt(Object javaObject) {
            if (javaObject instanceof Boolean) {
                return (Boolean) javaObject;
            }
            throw new PythonException("Cannot cast object of type " + javaObject.getClass().getName() + " to Boolean");
        }

        @Override
        public Boolean toJava(PythonObject pythonObject) {
            PythonGIL.assertThreadSafe();
            PyObject builtins = PyImport_ImportModule("builtins");
            PyObject boolF = PyObject_GetAttrString(builtins, "bool");

            PythonObject bool = new PythonObject(boolF, false).call(pythonObject);
            boolean ret = PyLong_AsLong(bool.getNativePythonObject()) > 0;
            bool.del();
            Py_DecRef(boolF);
            Py_DecRef(builtins);
            return ret;
        }

        @Override
        public PythonObject toPython(Boolean javaObject) {
            return new PythonObject(PyBool_FromLong(javaObject ? 1 : 0));
        }
    };


    public static final PythonType<List> LIST = new PythonType<List>("list", List.class) {

        @Override
        public List adapt(Object javaObject) {
            if (javaObject instanceof List) {
                return (List) javaObject;
            } else if (javaObject instanceof Object[]) {
                return Arrays.asList((Object[]) javaObject);
            } else {
                throw new PythonException("Cannot cast object of type " + javaObject.getClass().getName() + " to List");
            }
        }

        @Override
        public List toJava(PythonObject pythonObject) {
            PythonGIL.assertThreadSafe();
            List ret = new ArrayList();
            long n = PyObject_Size(pythonObject.getNativePythonObject());
            if (n < 0) {
                throw new PythonException("Object cannot be interpreted as a List");
            }
            for (long i = 0; i < n; i++) {
                PyObject pyIndex = PyLong_FromLong(i);
                PyObject pyItem = PyObject_GetItem(pythonObject.getNativePythonObject(),
                        pyIndex);
                Py_DecRef(pyIndex);
                PythonType pyItemType = getPythonTypeForPythonObject(new PythonObject(pyItem, false));
                ret.add(pyItemType.toJava(new PythonObject(pyItem, false)));
                Py_DecRef(pyItem);
            }
            return ret;
        }

        @Override
        public PythonObject toPython(List javaObject) {
            PythonGIL.assertThreadSafe();
            PyObject pyList = PyList_New(javaObject.size());
            for (int i = 0; i < javaObject.size(); i++) {
                Object item = javaObject.get(i);
                PythonObject pyItem;
                boolean owned;
                if (item instanceof PythonObject) {
                    pyItem = (PythonObject) item;
                    owned = false;
                } else if (item instanceof PyObject) {
                    pyItem = new PythonObject((PyObject) item, false);
                    owned = false;
                } else {
                    pyItem = PythonTypes.convert(item);
                    owned = true;
                }
                Py_IncRef(pyItem.getNativePythonObject()); // reference will be stolen by PyList_SetItem()
                PyList_SetItem(pyList, i, pyItem.getNativePythonObject());
                if (owned) pyItem.del();
            }
            return new PythonObject(pyList);
        }
    };

    public static final PythonType<Map> DICT = new PythonType<Map>("dict", Map.class) {

        @Override
        public Map adapt(Object javaObject) {
            if (javaObject instanceof Map) {
                return (Map) javaObject;
            }
            throw new PythonException("Cannot cast object of type " + javaObject.getClass().getName() + " to Map");
        }

        @Override
        public Map toJava(PythonObject pythonObject) {
            PythonGIL.assertThreadSafe();
            HashMap ret = new HashMap();
            PyObject dictType = new PyObject(PyDict_Type());
            if (PyObject_IsInstance(pythonObject.getNativePythonObject(), dictType) != 1) {
                throw new PythonException("Expected dict, received: " + pythonObject.toString());
            }

            PyObject keys = PyDict_Keys(pythonObject.getNativePythonObject());
            PyObject keysIter = PyObject_GetIter(keys);
            PyObject vals = PyDict_Values(pythonObject.getNativePythonObject());
            PyObject valsIter = PyObject_GetIter(vals);
            try {
                long n = PyObject_Size(pythonObject.getNativePythonObject());
                for (long i = 0; i < n; i++) {
                    PythonObject pyKey = new PythonObject(PyIter_Next(keysIter), false);
                    PythonObject pyVal = new PythonObject(PyIter_Next(valsIter), false);
                    PythonType pyKeyType = getPythonTypeForPythonObject(pyKey);
                    PythonType pyValType = getPythonTypeForPythonObject(pyVal);
                    ret.put(pyKeyType.toJava(pyKey), pyValType.toJava(pyVal));
                    Py_DecRef(pyKey.getNativePythonObject());
                    Py_DecRef(pyVal.getNativePythonObject());
                }
            } finally {
                Py_DecRef(keysIter);
                Py_DecRef(valsIter);
                Py_DecRef(keys);
                Py_DecRef(vals);
            }
            return ret;
        }

        @Override
        public PythonObject toPython(Map javaObject) {
            PythonGIL.assertThreadSafe();
            PyObject pyDict = PyDict_New();
            for (Object k : javaObject.keySet()) {
                PythonObject pyKey;
                if (k instanceof PythonObject) {
                    pyKey = (PythonObject) k;
                } else if (k instanceof PyObject) {
                    pyKey = new PythonObject((PyObject) k);
                } else {
                    pyKey = PythonTypes.convert(k);
                }
                Object v = javaObject.get(k);
                PythonObject pyVal;
                pyVal = PythonTypes.convert(v);
                int errCode = PyDict_SetItem(pyDict, pyKey.getNativePythonObject(), pyVal.getNativePythonObject());
                if (errCode != 0) {
                    String keyStr = pyKey.toString();
                    pyKey.del();
                    pyVal.del();
                    throw new PythonException("Unable to create python dictionary. Unhashable key: " + keyStr);
                }
                pyKey.del();
                pyVal.del();
            }
            return new PythonObject(pyDict);
        }
    };
}
