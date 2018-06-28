package org.nd4j.tensorflow.conversion;

import lombok.Getter;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import static org.bytedeco.javacpp.tensorflow.*;


public class TensorflowConversion {

    private   static Deallocator_Pointer_long_Pointer calling;


    public TensorflowConversion() {
        if(calling == null)
            calling = DummyDeAllocator.getInstance();

        TensorflowDeAllocatorHolder.addDeAllocatorRef(calling);
    }


    /**
     *
     * @param ndArray
     * @return
     */
    public TF_Tensor tensorFromNDArray(INDArray ndArray) {
        long[] ndShape = ndArray.shape();
        long[] tfShape = new long[ndShape.length];
        for (int i = 0; i < ndShape.length; i++) {
            tfShape[i] = ndShape[i];
        }

        int type;
        DataBuffer data = ndArray.data();
        DataBuffer.Type dataType = data.dataType();
        switch (dataType) {
            case DOUBLE: type = DT_DOUBLE; break;
            case FLOAT:  type = DT_FLOAT;  break;
            case INT:    type = DT_INT32;  break;
            case HALF:   type = DT_HALF;   break;
            case COMPRESSED:
                CompressedDataBuffer compressedData = (CompressedDataBuffer)data;
                CompressionDescriptor desc = compressedData.getCompressionDescriptor();
                String algo = desc.getCompressionAlgorithm();
                switch (algo) {
                    case "FLOAT16": type = DT_HALF;   break;
                    case "INT8":    type = DT_INT8;   break;
                    case "UINT8":   type = DT_UINT8;  break;
                    case "INT16":   type = DT_INT16;  break;
                    case "UINT16":  type = DT_UINT16; break;
                    default: throw new IllegalArgumentException("Unsupported compression algorithm: " + algo);
                }
                break;
            case LONG: type = DT_INT64; break;
            default: throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }

        try {
            Nd4j.getAffinityManager().ensureLocation(ndArray, AffinityManager.Location.HOST);
        } catch (Exception e) {
            // ND4J won't let us access compressed data in GPU memory, so we'll leave TensorFlow do the conversion instead
            ndArray.getDouble(0); // forces decompression and data copy to host
            data = ndArray.data();
            dataType = data.dataType();
            switch (dataType) {
                case DOUBLE: type = DT_DOUBLE; break;
                case FLOAT:  type = DT_FLOAT;  break;
                case INT:    type = DT_INT32;  break;
                case LONG:   type = DT_INT64;  break;
                default: throw new IllegalArgumentException("Unsupported data type: " + dataType);
            }
        }


        LongPointer longPointer = new LongPointer(tfShape);

        TF_Tensor tf_tensor = TF_NewTensor(
                type,
                longPointer,
                tfShape.length,
                aliasPointerForType(dataType,data.pointer()),
                data.length() * ndArray.data().getElementSize(),
                calling,null);

        return tf_tensor;

    }

    /**
     *
     * @param tensor
     * @return
     */
    public INDArray ndArrayFromTensor(TF_Tensor tensor) {
        int rank = TF_NumDims(tensor);

        int[] ndShape;
        if (rank == 0) {
            // scalar
            ndShape = new int[] { 1 };
        } else {
            ndShape = new int[rank];
            for (int i = 0; i < ndShape.length; i++) {
                ndShape[i] = (int) TF_Dim(tensor,i);
            }
        }

        int length = ArrayUtil.prod(ndShape);
        Pointer pointer = TF_TensorData(tensor);
        int tfType = TF_TensorType(tensor);
        DataBuffer.Type nd4jType = typeFor(tfType);
        Indexer indexer = indexerForType(nd4jType,pointer);
        DataBuffer d = Nd4j.createBuffer(indexer.pointer(),nd4jType,length,indexer);
        INDArray array = Nd4j.create(d,ndShape);
        Nd4j.getAffinityManager().tagLocation(array, AffinityManager.Location.HOST);
        return array;
    }


    public Pointer aliasPointerForType(DataBuffer.Type type,Pointer pointer) {
        switch(type) {
            case DOUBLE: new DoublePointer(pointer);
            case FLOAT: return new FloatPointer(pointer);
            case INT: return new IntPointer(pointer);
            case LONG: return new LongPointer(pointer);
            default: throw new IllegalArgumentException("Illegal type " + type);
        }
    }

    public Indexer indexerForType(DataBuffer.Type type,Pointer pointer) {
        switch(type) {
            case DOUBLE: return DoubleIndexer.create(new DoublePointer(pointer));
            case FLOAT: return FloatIndexer.create(new FloatPointer(pointer));
            case INT: return IntIndexer.create(new IntPointer(pointer));
            case LONG: return LongIndexer.create(new LongPointer(pointer));
            default: throw new IllegalArgumentException("Illegal type " + type);
        }
    }

    public DataBuffer.Type typeFor(int tensorflowType) {
        switch(tensorflowType) {
            case DT_DOUBLE: return DataBuffer.Type.DOUBLE;
            case DT_FLOAT: return DataBuffer.Type.FLOAT;
            case DT_INT8: return DataBuffer.Type.INT;
            case DT_INT16: return DataBuffer.Type.LONG;
            default: throw new IllegalArgumentException("Illlegal type " + tensorflowType);
        }
    }

}
