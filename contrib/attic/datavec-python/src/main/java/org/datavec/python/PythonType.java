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

package org.datavec.python;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.datavec.python.Python.importModule;


/**
 *
 * @param <T> Corresponding Java type for the Python type
 */
public abstract class PythonType<T> {

    public abstract T toJava(PythonObject pythonObject) throws PythonException;
    private final TypeName typeName;

    public enum TypeName{
        STR,
        INT,
        FLOAT,
        BOOL,
        LIST,
        DICT,
        NDARRAY,
        BYTES
    }
    private PythonType(TypeName typeName){
        this.typeName = typeName;
    }
    public TypeName getName(){return typeName;}
    public String toString(){
        return getName().name();
    }
    public static PythonType valueOf(String typeName) throws PythonException{
        try{
            typeName.valueOf(typeName);
        } catch (IllegalArgumentException iae){
            throw new PythonException("Invalid python type: " + typeName, iae);
        }
        try{
            return (PythonType)PythonType.class.getField(typeName).get(null); // shouldn't fail
        } catch (Exception e){
            throw new RuntimeException(e);
        }

    }
    public static PythonType valueOf(TypeName typeName){
        try{
            return valueOf(typeName.name()); // shouldn't fail
        }catch (PythonException pe){
            throw new RuntimeException(pe);
        }
    }

    /**
     * Since multiple java types can map to the same python type,
     * this method "normalizes" all supported incoming objects to T
     *
     * @param object object to be converted to type T
     * @return
     */
    public T convert(Object object) throws PythonException {
        return (T) object;
    }

    public static final PythonType<String> STR = new PythonType<String>(TypeName.STR) {
        @Override
        public String toJava(PythonObject pythonObject) throws PythonException {
            if (!Python.isinstance(pythonObject, Python.strType())) {
                throw new PythonException("Expected variable to be str, but was " + Python.type(pythonObject));
            }
            return pythonObject.toString();
        }

        @Override
        public String convert(Object object) {
            return object.toString();
        }
    };

    public static final PythonType<Long> INT = new PythonType<Long>(TypeName.INT) {
        @Override
        public Long toJava(PythonObject pythonObject) throws PythonException {
            if (!Python.isinstance(pythonObject, Python.intType())) {
                throw new PythonException("Expected variable to be int, but was " + Python.type(pythonObject));
            }
            return pythonObject.toLong();
        }

        @Override
        public Long convert(Object object) throws PythonException {
            if (object instanceof Number) {
                return ((Number) object).longValue();
            }
            throw new PythonException("Unable to cast " + object + " to Long.");
        }
    };

    public static final PythonType<Double> FLOAT = new PythonType<Double>(TypeName.FLOAT) {
        @Override
        public Double toJava(PythonObject pythonObject) throws PythonException {
            if (!Python.isinstance(pythonObject, Python.floatType())) {
                throw new PythonException("Expected variable to be float, but was " + Python.type(pythonObject));
            }
            return pythonObject.toDouble();
        }

        @Override
        public Double convert(Object object) throws PythonException {
            if (object instanceof Number) {
                return ((Number) object).doubleValue();
            }
            throw new PythonException("Unable to cast " + object + " to Double.");
        }
    };

    public static final PythonType<Boolean> BOOL = new PythonType<Boolean>(TypeName.BOOL) {
        @Override
        public Boolean toJava(PythonObject pythonObject) throws PythonException {
            if (!Python.isinstance(pythonObject, Python.boolType())) {
                throw new PythonException("Expected variable to be bool, but was " + Python.type(pythonObject));
            }
            return pythonObject.toBoolean();
        }

        @Override
        public Boolean convert(Object object) throws PythonException {
            if (object instanceof Number) {
                return ((Number) object).intValue() != 0;
            } else if (object instanceof Boolean) {
                return (Boolean) object;
            }
            throw new PythonException("Unable to cast " + object + " to Boolean.");
        }
    };

    public static final PythonType<List> LIST = new PythonType<List>(TypeName.LIST) {
        @Override
        public List toJava(PythonObject pythonObject) throws PythonException {
            if (!Python.isinstance(pythonObject, Python.listType())) {
                throw new PythonException("Expected variable to be list, but was " + Python.type(pythonObject));
            }
            return pythonObject.toList();
        }

        @Override
        public List convert(Object object) throws PythonException {
            if (object instanceof java.util.List) {
                return (List) object;
            } else if (object instanceof org.json.JSONArray) {
                org.json.JSONArray jsonArray = (org.json.JSONArray) object;
                return jsonArray.toList();

            } else if (object instanceof Object[]) {
                return Arrays.asList((Object[]) object);
            }
            throw new PythonException("Unable to cast " + object + " to List.");
        }
    };

    public static final PythonType<Map> DICT = new PythonType<Map>(TypeName.DICT) {
        @Override
        public Map toJava(PythonObject pythonObject) throws PythonException {
            if (!Python.isinstance(pythonObject, Python.dictType())) {
                throw new PythonException("Expected variable to be dict, but was " + Python.type(pythonObject));
            }
            return pythonObject.toMap();
        }

        @Override
        public Map convert(Object object) throws PythonException {
            if (object instanceof Map) {
                return (Map) object;
            }
            throw new PythonException("Unable to cast " + object + " to Map.");
        }
    };

    public static final PythonType<INDArray> NDARRAY = new PythonType<INDArray>(TypeName.NDARRAY) {
        @Override
        public INDArray toJava(PythonObject pythonObject) throws PythonException {
            PythonObject np = importModule("numpy");
            if (!Python.isinstance(pythonObject, np.attr("ndarray"), np.attr("generic"))) {
                throw new PythonException("Expected variable to be numpy.ndarray, but was " + Python.type(pythonObject));
            }
            return pythonObject.toNumpy().getNd4jArray();
        }

        @Override
        public INDArray convert(Object object) throws PythonException {
            if (object instanceof INDArray) {
                return (INDArray) object;
            } else if (object instanceof NumpyArray) {
                return ((NumpyArray) object).getNd4jArray();
            }
            throw new PythonException("Unable to cast " + object + " to INDArray.");
        }
    };

    public static final PythonType<BytePointer> BYTES = new PythonType<BytePointer>(TypeName.BYTES) {
        @Override
        public BytePointer toJava(PythonObject pythonObject) throws PythonException {
            return pythonObject.toBytePointer();
        }

        @Override
        public BytePointer convert(Object object) throws PythonException {
            if (object instanceof BytePointer) {
                return (BytePointer) object;
            } else if (object instanceof Pointer) {
                return new BytePointer((Pointer) object);
            }
            throw new PythonException("Unable to cast " + object + " to BytePointer.");
        }
    };
}
