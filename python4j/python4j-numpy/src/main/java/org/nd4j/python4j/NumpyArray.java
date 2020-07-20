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


package org.nd4j.python4j;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.cpython.PyObject;
import org.bytedeco.cpython.PyTypeObject;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.numpy.PyArrayObject;
import org.bytedeco.numpy.global.numpy;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.cpython.global.python.Py_DecRef;
import static org.bytedeco.numpy.global.numpy.*;
import static org.bytedeco.numpy.global.numpy.NPY_ARRAY_CARRAY;
import static org.bytedeco.numpy.global.numpy.PyArray_Type;

@Slf4j
public class NumpyArray extends PythonType<INDArray> {

    public static final NumpyArray INSTANCE;
    private static final AtomicBoolean init = new AtomicBoolean(false);
    private static final Map<String, DataBuffer> cache = new HashMap<>();

    static {
        new PythonExecutioner();
        INSTANCE = new NumpyArray();
    }

    @Override
    public File[] packages(){
        try{
            return new File[]{numpy.cachePackage()};
        }catch(Exception e){
            throw new PythonException(e);
        }

    }

    public synchronized void init() {
        if (init.get()) return;
        init.set(true);
        if (PythonGIL.locked()) {
            throw new PythonException("Can not initialize numpy - GIL already acquired.");
        }
        int err = numpy._import_array();
        if (err < 0){
            System.out.println("Numpy import failed!");
            throw new PythonException("Numpy import failed!");
        }
    }

    public NumpyArray() {
        super("numpy.ndarray", INDArray.class);

    }

    @Override
    public INDArray toJava(PythonObject pythonObject) {
        log.info("Converting PythonObject to INDArray...");
        PyObject np = PyImport_ImportModule("numpy");
        PyObject ndarray = PyObject_GetAttrString(np, "ndarray");
        if (PyObject_IsInstance(pythonObject.getNativePythonObject(), ndarray) != 1) {
            Py_DecRef(ndarray);
            Py_DecRef(np);
            throw new PythonException("Object is not a numpy array! Use Python.ndarray() to convert object to a numpy array.");
        }
        Py_DecRef(ndarray);
        Py_DecRef(np);
        PyArrayObject npArr = new PyArrayObject(pythonObject.getNativePythonObject());
        long[] shape = new long[PyArray_NDIM(npArr)];
        SizeTPointer shapePtr = PyArray_SHAPE(npArr);
        if (shapePtr != null)
            shapePtr.get(shape, 0, shape.length);
        long[] strides = new long[shape.length];
        SizeTPointer stridesPtr = PyArray_STRIDES(npArr);
        if (stridesPtr != null)
            stridesPtr.get(strides, 0, strides.length);
        int npdtype = PyArray_TYPE(npArr);

        DataType dtype;
        switch (npdtype) {
            case NPY_DOUBLE:
                dtype = DataType.DOUBLE;
                break;
            case NPY_FLOAT:
                dtype = DataType.FLOAT;
                break;
            case NPY_SHORT:
                dtype = DataType.SHORT;
                break;
            case NPY_INT:
                dtype = DataType.INT32;
                break;
            case NPY_LONG:
                dtype = DataType.INT64;
                break;
            case NPY_UINT:
                dtype = DataType.UINT32;
                break;
            case NPY_BYTE:
                dtype = DataType.INT8;
                break;
            case NPY_UBYTE:
                dtype = DataType.UINT8;
                break;
            case NPY_BOOL:
                dtype = DataType.BOOL;
                break;
            case NPY_HALF:
                dtype = DataType.FLOAT16;
                break;
            case NPY_LONGLONG:
                dtype = DataType.INT64;
                break;
            case NPY_USHORT:
                dtype = DataType.UINT16;
                break;
            case NPY_ULONG:
            case NPY_ULONGLONG:
                dtype = DataType.UINT64;
                break;
            default:
                throw new PythonException("Unsupported array data type: " + npdtype);
        }
        long size = 1;
        for (int i = 0; i < shape.length; size *= shape[i++]) ;

        INDArray ret;
        long address = PyArray_DATA(npArr).address();
        String key = address + "_" + size + "_" + dtype;
        DataBuffer buff = cache.get(key);
        if (buff == null) {
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                Pointer ptr = NativeOpsHolder.getInstance().getDeviceNativeOps().pointerForAddress(address);
                ptr = ptr.limit(size);
                ptr = ptr.capacity(size);
                buff = Nd4j.createBuffer(ptr, size, dtype);
                cache.put(key, buff);
            }
        }
        int elemSize = buff.getElementSize();
        long[] nd4jStrides = new long[strides.length];
        for (int i = 0; i < strides.length; i++) {
            nd4jStrides[i] = strides[i] / elemSize;
        }
        ret = Nd4j.create(buff, shape, nd4jStrides, 0, Shape.getOrder(shape, nd4jStrides, 1), dtype);
        Nd4j.getAffinityManager().tagLocation(ret, AffinityManager.Location.HOST);
        log.info("Done.");
        return ret;


    }

    @Override
    public PythonObject toPython(INDArray indArray) {
        log.info("Converting INDArray to PythonObject...");
        DataType dataType = indArray.dataType();
        DataBuffer buff = indArray.data();
        String key = buff.pointer().address() + "_" + buff.length() + "_" + dataType;
        cache.put(key, buff);
        int numpyType;
        String ctype;
        switch (dataType) {
            case DOUBLE:
                numpyType = NPY_DOUBLE;
                ctype = "c_double";
                break;
            case FLOAT:
            case BFLOAT16:
                numpyType = NPY_FLOAT;
                ctype = "c_float";
                break;
            case SHORT:
                numpyType = NPY_SHORT;
                ctype = "c_short";
                break;
            case INT:
                numpyType = NPY_INT;
                ctype = "c_int";
                break;
            case LONG:
                numpyType = NPY_INT64;
                ctype = "c_int64";
                break;
            case UINT16:
                numpyType = NPY_USHORT;
                ctype = "c_uint16";
                break;
            case UINT32:
                numpyType = NPY_UINT;
                ctype = "c_uint";
                break;
            case UINT64:
                numpyType = NPY_UINT64;
                ctype = "c_uint64";
                break;
            case BOOL:
                numpyType = NPY_BOOL;
                ctype = "c_bool";
                break;
            case BYTE:
                numpyType = NPY_BYTE;
                ctype = "c_byte";
                break;
            case UBYTE:
                numpyType = NPY_UBYTE;
                ctype = "c_ubyte";
                break;
            case HALF:
                numpyType = NPY_HALF;
                ctype = "c_short";
                break;
            default:
                throw new RuntimeException("Unsupported dtype: " + dataType);
        }

        long[] shape = indArray.shape();
        INDArray inputArray = indArray;
        if (dataType == DataType.BFLOAT16) {
            log.warn("Creating copy of array as bfloat16 is not supported by numpy.");
            inputArray = indArray.castTo(DataType.FLOAT);
        }

        //Sync to host memory in the case of CUDA, before passing the host memory pointer to Python

        Nd4j.getAffinityManager().ensureLocation(inputArray, AffinityManager.Location.HOST);

        // PyArray_Type() call causes jvm crash in linux cpu if GIL is acquired by non main thread.
        // Using Interpreter for now:

//        try(PythonContextManager.Context context = new PythonContextManager.Context("__np_array_converter")){
//            log.info("Stringing exec...");
//            String code = "import ctypes\nimport numpy as np\n" +
//                    "cArr = (ctypes." + ctype + "*" + indArray.length() + ")"+
//                    ".from_address(" + indArray.data().pointer().address() + ")\n"+
//                    "npArr = np.frombuffer(cArr, dtype=" + ((numpyType == NPY_HALF) ? "'half'" : "ctypes." + ctype)+
//                    ").reshape(" + Arrays.toString(indArray.shape()) + ")";
//            PythonExecutioner.exec(code);
//            log.info("exec done.");
//            PythonObject ret = PythonExecutioner.getVariable("npArr");
//            Py_IncRef(ret.getNativePythonObject());
//            return ret;
//
//        }
        log.info("NUMPY: PyArray_Type()");
        PyTypeObject pyTypeObject = PyArray_Type();


        log.info("NUMPY: PyArray_New()");
        PyObject npArr = PyArray_New(pyTypeObject, shape.length, new SizeTPointer(shape),
                numpyType, null,
                inputArray.data().addressPointer(),
                0, NPY_ARRAY_CARRAY, null);
        log.info("Done.");
        return new PythonObject(npArr);
    }

    @Override
    public boolean accepts(Object javaObject) {
        return javaObject instanceof INDArray;
    }

    @Override
    public INDArray adapt(Object javaObject) {
        if (javaObject instanceof INDArray) {
            return (INDArray) javaObject;
        }
        throw new PythonException("Cannot cast object of type " + javaObject.getClass().getName() + " to INDArray");
    }
}
