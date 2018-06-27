package org.nd4j.tensorflow.conversion;

import lombok.Getter;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.BasicNDArrayCompressor;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ReflectionUtils;
import org.nd4j.linalg.util.ArrayUtil;


import java.lang.ref.PhantomReference;
import java.lang.reflect.Field;
import java.nio.Buffer;

import static org.bytedeco.javacpp.tensorflow.*;


public class TensorflowConversion {
    private static   Deallocator_Pointer_long_Pointer calling = new Deallocator_Pointer_long_Pointer() {
        public void call(Pointer data, long length) {
            System.out.println("calling");
        }
    };;;
    private String[] inputNames,outputNames;

    private tensorflow.GraphDef castGraph;
    private Session castSession;
    private Session session;


    private BasicNDArrayCompressor compressor = BasicNDArrayCompressor.getInstance();


    @Getter private  tensorflow.GraphDef graph;
    @Getter private  byte[] graphDef;

    private static Field deallocatorField;

    private volatile boolean isClosed = false;

    /**
     * Fraction of GPU memory to let TensorFlow hoard for the whole process.
     */
    @Getter
    private double gpuMemoryFraction = 0.5;

    /**
     * If set to false, TensorFlow grabs off the bat as much as specified by {@link #gpuMemoryFraction}.
     */
    @Getter private boolean gpuMemoryAllowGrowth = true;

    /**
     * A string like "/cpu:0", "/device:GPU:0", "/device:GPU:1", etc.
     * Only works if the device is not already set on the nodes of the graph.
     */
    @Getter private String defaultDevice = null;

    /**
     * If true, logs detailed information about which devices are allocated.
     */
    @Getter private boolean logDevicePlacement = false;



    Session openSession(GraphDef def) {
        SetDefaultDevice(defaultDevice, def);
        ConfigProto configProto = new ConfigProto();
        configProto.set_log_device_placement(logDevicePlacement);
        configProto.gpu_options().set_per_process_gpu_memory_fraction(gpuMemoryFraction);
        configProto.gpu_options().set_allow_growth(gpuMemoryAllowGrowth);
        SessionOptions options = new SessionOptions();
        options.config(configProto);
        Session sess = new Session(options);
        Status s = sess.Create(def);
        if (!s.ok()) {
            throw new RuntimeException(s.error_message().getString());
        }
        return sess;
    }

    Tensor[] runSession(Session sess, String[] inputNames, Tensor[] inputTensors, String[] outputNames) {
        TensorVector outputTensors = new TensorVector();
        Status s = sess.Run(new StringTensorPairVector(inputNames, inputTensors),
                new StringVector(outputNames), new StringVector(), outputTensors);
        if (!s.ok()) {
            throw new RuntimeException(s.error_message().getString());
        }
        int n = (int)outputTensors.size();
        Tensor[] outputs = new Tensor[n];
        for (int i = 0; i < n; i++) {
            outputs[i] = new Tensor(outputTensors.get(i));
        }
        return outputs;
    }

    Tensor[] castTensors(Tensor... inputTensors) {
        String[] inNames = new String[inputNames.length];
        String[] outNames = new String[inputNames.length];
        for (int i = 0; i < inputTensors.length; i++) {
            inNames[i] = "in_" + i;
            outNames[i] = "out_" + i;
        }
        if (castGraph == null || castSession == null) {
            synchronized (this) {
                if (castGraph == null || castSession == null) {
                    Scope root = Scope.NewRootScope();
                    int n = graph.node_size();
                    for (int i = 0; i < inputTensors.length; i++) {
                        for (int j = 0; j < n; j++) {
                            NodeDef node = graph.node(j);
                            if (node.name().getString().equals(inputNames[i])) {
                                AttrValue attr = node.attr().get(new BytePointer("dtype"));
                                Placeholder p = new Placeholder(root.WithOpName(inNames[i]), inputTensors[i].dtype());
                                new CastOp(root.WithOpName(outNames[i]), p.asInput(), attr.type());
                            }
                        }
                    }
                    castGraph = new GraphDef();
                    Status s = root.ToGraphDef(castGraph);
                    if (!s.ok()) {
                        throw new RuntimeException(s.error_message().getString());
                    }
                    castSession = openSession(castGraph);
                }
            }
        }
        return runSession(castSession, inNames, inputTensors, outNames);
    }

    /** Adjust the data type of the inputs to match the model. */
    INDArray[] maybeCompress(INDArray... inputs) {
        INDArray[] inputs2 = new INDArray[inputs.length];
        int n = graph.node_size();
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < n; j++) {
                NodeDef node = graph.node(j);
                if (node.name().getString().equals(inputNames[i])) {
                    AttrValue attr = node.attr().get(new BytePointer("dtype"));
                    String compressedAlgo = "", recompressAlgo = "";
                    if (inputs[i].data().dataType() == DataBuffer.Type.COMPRESSED) {
                        CompressedDataBuffer compressedData = (CompressedDataBuffer)inputs[i].data();
                        CompressionDescriptor desc = compressedData.getCompressionDescriptor();
                        compressedAlgo = desc.getCompressionAlgorithm();
                    }
                    switch (attr.type()) {
                        case DT_FLOAT:
                        case DT_DOUBLE:
                        case DT_INT32:
                        case DT_UINT32:
                        case DT_INT64:
                        case DT_UINT64:
                            recompressAlgo = "";
                            break;
                        case DT_INT8:
                            if (!compressedAlgo.equals("INT8")) {
                                recompressAlgo = "INT8";
                            }
                            break;
                        case DT_UINT8:
                            if (!compressedAlgo.equals("UINT8")) {
                                recompressAlgo = "UINT8";
                            }
                            break;
                        case DT_INT16:
                            if (!compressedAlgo.equals("INT16")) {
                                recompressAlgo = "INT16";
                            }
                            break;
                        case DT_UINT16:
                            // TODO: use "UINT16" when it becomes supported
                            if (!compressedAlgo.equals("INT16")) {
                                recompressAlgo = "INT16";
                            }
                            break;
                        case DT_BOOL:
                            if (!compressedAlgo.equals("THRESHOLD")) {
                                recompressAlgo = "THRESHOLD";
                            }
                            break;
                        case DT_HALF:
                            if (!compressedAlgo.equals("FLOAT16")) {
                                recompressAlgo = "FLOAT16";
                            }
                            break;
                        default: throw new IllegalArgumentException("Unsupported dtype: " + attr.type());
                    }
                    inputs2[i] = inputs[i];
                    compressor.decompressi(inputs2[i]);
                    if (recompressAlgo.length() > 0) {
                        if (inputs2[i].isView()) {
                            // TODO: remove when support for views is added to ND4J compression
                            inputs2[i] = inputs2[i].dup();
                        }
                        compressor.compressi(inputs2[i], recompressAlgo);
                    }
                    break;
                }
            }
        }
        return inputs2;
    }



    public static TF_Tensor tensorFromNDArray(INDArray ndArray) throws NoSuchFieldException {
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
                ndArray.data().pointer(),
                ndArray.data().length() * ndArray.data().getElementSize(),
                calling,null);

        return tf_tensor;
    }

    public static INDArray ndArrayFromTensor(TF_Tensor tensor) {
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
        DataBuffer d = Nd4j.createBuffer(indexer.pointer(),nd4jType,length,indexerForType(nd4jType,pointer));
        INDArray array = Nd4j.create(d,ndShape);
        Nd4j.getAffinityManager().tagLocation(array, AffinityManager.Location.HOST);
        return array;
    }


    public static Pointer aliasPointerForType(DataBuffer.Type type,Pointer pointer) {
        switch(type) {
            case DOUBLE: new DoublePointer(pointer);
            case FLOAT: return new FloatPointer(pointer);
            case INT: return new IntPointer(pointer);
            case LONG: return new LongPointer(pointer);
            default: throw new IllegalArgumentException("Illegal type " + type);
        }
    }

    public static Indexer indexerForType(DataBuffer.Type type,Pointer pointer) {
        switch(type) {
            case DOUBLE: return DoubleIndexer.create(new DoublePointer(pointer));
            case FLOAT: return FloatIndexer.create(new FloatPointer(pointer));
            case INT: return IntIndexer.create(new IntPointer(pointer));
            case LONG: return LongIndexer.create(new LongPointer(pointer));
            default: throw new IllegalArgumentException("Illegal type " + type);
        }
    }

    public static DataBuffer.Type typeFor(int tensorflowType) {
        switch(tensorflowType) {
            case DT_DOUBLE: return DataBuffer.Type.DOUBLE;
            case DT_FLOAT: return DataBuffer.Type.FLOAT;
            case DT_INT8: return DataBuffer.Type.INT;
            case DT_INT16: return DataBuffer.Type.LONG;
            default: throw new IllegalArgumentException("Illlegal type " + tensorflowType);
        }
    }

}
