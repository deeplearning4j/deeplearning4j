package org.nd4j.compression;

import lombok.NonNull;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.NDArrayCompressor;
import org.reflections.Reflections;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
public class DataCompressor {
    private static final DataCompressor INSTANCE = new DataCompressor();

    protected Map<String, NDArrayCompressor> codecs;

    protected String defaultCompression = "FP16";

    private DataCompressor() {
        loadCompressors();
    }

    protected void loadCompressors() {
        /*
            We scan classpath for NDArrayCompressor implementations and add them one by one to codecs map
         */
        codecs = new ConcurrentHashMap<>();
        Reflections reflections = new Reflections("org.nd4j");
        Set<Class<? extends NDArrayCompressor>> classes = reflections.getSubTypesOf(NDArrayCompressor.class);
        for (Class<? extends NDArrayCompressor> impl : classes) {
            try {
                NDArrayCompressor compressor = impl.newInstance();


                System.out.println("Adding compressor: " + compressor.getDescriptor().toUpperCase());

                codecs.put(compressor.getDescriptor().toUpperCase(), compressor);
            } catch (InstantiationException i) {
                ; // we need catch there, to avoid exceptions at abstract classes
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    public void printAvailableCompressors() {
        StringBuilder builder = new StringBuilder();
        builder.append("Available compressors: ");
        for (String comp : codecs.keySet()) {
            builder.append("[").append(comp).append("] ");
        }

        System.out.println(builder.toString());
    }

    public static DataCompressor getInstance() {
        return INSTANCE;
    }

    public void setDefaultCompression(@NonNull String algorithm) {
        algorithm = algorithm.toUpperCase();
 //       if (!codecs.containsKey(algorithm))
//            throw new RuntimeException("Non-existent compression algorithm requested: [" + algorithm + "]");

        synchronized (this) {
            defaultCompression = algorithm;
        }
    }

    public String getDefaultCompression() {
        synchronized (this) {
            return defaultCompression;
        }
    }

    public DataBuffer compress(DataBuffer buffer) {
        return compress(buffer, getDefaultCompression());
    }

    public DataBuffer compress(DataBuffer buffer, String algorithm) {
        algorithm = algorithm.toUpperCase();
        if (!codecs.containsKey(algorithm))
            throw new RuntimeException("Non-existent compression algorithm requested: [" + algorithm + "]");

        return codecs.get(algorithm).compress(buffer);
    }

    public INDArray compress(INDArray array) {
        return compress(array, getDefaultCompression());
    }

    public INDArray compress(INDArray array, String algorithm) {
        algorithm = algorithm.toUpperCase();
        if (!codecs.containsKey(algorithm))
            throw new RuntimeException("Non-existent compression algorithm requested: [" + algorithm + "]");

        return codecs.get(algorithm).compress(array);
    }

    public DataBuffer decompress(DataBuffer buffer) {
        if (buffer.dataType() != DataBuffer.Type.COMPRESSED)
            throw new IllegalStateException("You can't decompress DataBuffer with dataType of: " + buffer.dataType());

        CompressedDataBuffer comp = (CompressedDataBuffer) buffer;
        CompressionDescriptor descriptor = comp.getCompressionDescriptor();

        if (!codecs.containsKey(descriptor.getCompressionAlgorithm()))
            throw new RuntimeException("Non-existent compression algorithm requested: [" + descriptor.getCompressionAlgorithm() + "]");

        return codecs.get(descriptor.getCompressionAlgorithm()).decompress(buffer);
    }
}
