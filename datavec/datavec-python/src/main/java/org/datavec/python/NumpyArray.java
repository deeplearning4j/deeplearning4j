package org.datavec.python;

import org.apache.commons.lang3.ArrayUtils;
import org.bytedeco.javacpp.Pointer;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.DoublePointer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.linalg.api.buffer.DataType;


import java.util.ArrayList;


public class NumpyArray {

    private static NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private long address;
    private long[] shape;
    private long[] strides;
    private DataType dtype = DataType.FLOAT;
    private INDArray nd4jArray;

    public NumpyArray(long address, long[] shape, long strides[], boolean copy){
        this.address = address;
        this.shape = shape;
        this.strides = strides;
        setND4JArray();
        if (copy){
            nd4jArray = nd4jArray.dup();
            this.address = nd4jArray.data().address();

        }
    }
    public NumpyArray(long address, long[] shape, long strides[]){
        this(address, shape, strides, false);
    }

    public NumpyArray(long address, long[] shape, long strides[], DataType dtype){
        this(address, shape, strides, dtype, false);
    }

    public NumpyArray(long address, long[] shape, long strides[], DataType dtype, boolean copy){
        this.address = address;
        this.shape = shape;
        this.strides = strides;
        this.dtype = dtype;
        setND4JArray();
        if (copy){
            nd4jArray = nd4jArray.dup();
            this.address = nd4jArray.data().address();
        }
    }

    public long getAddress() {
        return address;
    }

    public long[] getShape() {
        return shape;
    }

    public long[] getStrides() {
        return strides;
    }

    public DataType getDType() {
        return dtype;
    }

    public JSONObject toJSON(){
        JSONObject jsonObject = new JSONObject();
        JSONArray data = new JSONArray();
        long size = 1;
        for (long d: nd4jArray.shape()){
            size *= d;
        }
        if (dtype == DataType.SHORT || dtype == DataType.INT || dtype == DataType.LONG){
            for (int i=0; i < size; i++){
                data.add((long)nd4jArray.getDouble(i));
            }
        }
        else if (dtype == DataType.FLOAT || dtype == DataType.DOUBLE){
            for (int i=0; i<size; i++){
                data.add(nd4jArray.getDouble(i));
            }
        }
        JSONArray shape = new JSONArray();
        for (long d: this.shape){
            shape.add(d);
        }
        String dtypeStr = dtype.toString();
        jsonObject.put("data", data);
        jsonObject.put("shape", shape);
        jsonObject.put("dtype", dtypeStr);
        return jsonObject;
    }


    public NumpyArray(JSONObject json) throws Exception{
        address = (Long)json.get("address");
        JSONArray shapeJson = (JSONArray)json.get("shape");
        shape = new long[shapeJson.size()];
        for (int i=0; i<shape.length; i++){
            shape[i] = (Long)shapeJson.get(i);
        }
        JSONArray stridesJson = (JSONArray)json.get("strides");
        strides = new long[stridesJson.size()];
        for (int i=0; i<strides.length; i++){
            strides[i] = (Long)stridesJson.get(i);
        }
        String dtypeStr = (String)json.get("dtype");
        if (dtypeStr != null){
            if (dtypeStr.equals("FLOAT")){
                dtype = DataType.FLOAT;
            }
            else if (dtypeStr.equals("DOUBLE")){
                dtype = DataType.DOUBLE;
            }
            else if (dtypeStr.equals("SHORT")){
                dtype = DataType.SHORT;
            }
            else if (dtypeStr.equals("INT")){
                dtype = DataType.INT;
            }
            else if (dtypeStr.equals("LONG")){
                dtype = DataType.LONG;
            }
            else{
                throw new Exception("Unsupported type: " + dtypeStr);
            }
        }
        setND4JArray();
    }

    public INDArray getND4JArray() {
        return nd4jArray;
    }

    private void setND4JArray(){
        long size = 1;
        for(long d: shape){
            size *= d;
        }
        Pointer ptr = nativeOps.pointerForAddress(address);
        ptr = ptr.limit(size);
        ptr = ptr.capacity(size);
        DataBuffer buff = Nd4j.createBuffer(ptr, size, dtype);
        int elemSize = buff.getElementSize();
        long[] nd4jStrides = new long[strides.length];
        for (int i=0; i<strides.length; i++){
            nd4jStrides[i] = strides[i] / elemSize;
        }
        this.nd4jArray = Nd4j.create(buff, shape, nd4jStrides, 0, 'c', dtype);

    }

    public NumpyArray(INDArray nd4jArray){
        DataBuffer buff = nd4jArray.data();
        address = buff.pointer().address();
        shape = nd4jArray.shape();
        long[] nd4jStrides = nd4jArray.stride();
        strides = new long[nd4jStrides.length];
        int elemSize = buff.getElementSize();
        for(int i=0; i<strides.length; i++){
            strides[i] = nd4jStrides[i] * elemSize;
        }
        dtype = nd4jArray.dataType();
        this.nd4jArray = nd4jArray;
    }

}
