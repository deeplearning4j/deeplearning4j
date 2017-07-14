package org.deeplearning4j.nn.conf.memory;

import lombok.NonNull;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.buffer.DataBuffer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Created by Alex on 13/07/2017.
 */

public class NetworkMemoryReport extends MemoryReport {

    private final Map<String, MemoryReport> layerAndVertexReports;
    private final Class<?> modelClass;
    private final String modelName;
    private final InputType[] networkInputTypes;

    public NetworkMemoryReport(@NonNull Map<String, MemoryReport> layerAndVertexReports,
                               @NonNull Class<?> modelClass,
                               String modelName,
                               @NonNull InputType... networkInputTypes) {
        this.layerAndVertexReports = layerAndVertexReports;
        this.modelClass = modelClass;
        this.modelName = modelName;
        this.networkInputTypes = networkInputTypes;
    }


    @Override
    public Class<?> getReportClass() {
        return modelClass;
    }

    @Override
    public String getName() {
        return modelName;
    }

    @Override
    public long getMemoryBytes(MemoryType memoryType, int minibatchSize, MemoryUseMode memoryUseMode,
                               CacheMode cacheMode, DataBuffer.Type dataType) {
        long totalBytes = 0;
        for (MemoryReport lmr : layerAndVertexReports.values()) {

            totalBytes += lmr.getMemoryBytes(memoryType, minibatchSize, memoryUseMode, cacheMode, dataType);
        }

        return totalBytes;
    }

    @Override
    public String toString() {

        long fixedMemBytes = getTotalMemoryBytes(0, MemoryUseMode.INFERENCE, CacheMode.NONE, DataBuffer.Type.FLOAT);
        long perEx = getTotalMemoryBytes(1, MemoryUseMode.INFERENCE, CacheMode.NONE, DataBuffer.Type.FLOAT) - fixedMemBytes;

        long fixedMemBytesTrain = getTotalMemoryBytes(0, MemoryUseMode.TRAINING, CacheMode.NONE, DataBuffer.Type.FLOAT);
        long perExTrain = getTotalMemoryBytes(1, MemoryUseMode.TRAINING, CacheMode.NONE, DataBuffer.Type.FLOAT) - fixedMemBytesTrain;

        Map<Class<?>, Integer> layerCounts = new LinkedHashMap<>();
        for (MemoryReport mr : layerAndVertexReports.values()) {
            if (layerCounts.containsKey(mr.getReportClass())) {
                layerCounts.put(mr.getReportClass(), layerCounts.get(mr.getReportClass()) + 1);
            } else {
                layerCounts.put(mr.getReportClass(), 1);
            }
        }

        StringBuilder sbLayerCounts = new StringBuilder();
        for (Map.Entry<Class<?>, Integer> e : layerCounts.entrySet()) {
            sbLayerCounts.append(e.getValue()).append(" x ").append(e.getKey().getSimpleName()).append(", ");
        }

        StringBuilder sb = new StringBuilder();
        sb.append("----- Network Memory Report -----\n")
                .append("  Model Class:                        ").append(modelClass.getName()).append("\n")
                .append("  Model Name:                         ").append(modelName).append("\n")
                .append("  Network Input:                      ").append(Arrays.toString(networkInputTypes)).append("\n")
                .append("  # Layers:                           ").append(layerAndVertexReports.size()).append("\n")
                .append("  Layer Types:                        ").append(sbLayerCounts).append("\n");

        appendFixedPlusVariable(sb, "  Inference Memory (FP32)             ", fixedMemBytes, perEx);
        appendFixedPlusVariable(sb, "  Training Memory (FP32):             ", fixedMemBytesTrain, perExTrain);

                sb.append("  Inference Memory Breakdown (FP32):\n");
        appendBreakDown(sb, MemoryUseMode.INFERENCE, CacheMode.NONE, DataBuffer.Type.FLOAT);

        sb.append("  Training Memory Breakdown (CacheMode = ").append(CacheMode.NONE).append(", FP32):\n");
        appendBreakDown(sb, MemoryUseMode.TRAINING, CacheMode.NONE, DataBuffer.Type.FLOAT);


        return sb.toString();
    }

    private void appendBreakDown(StringBuilder sb, MemoryUseMode useMode, CacheMode cacheMode, DataBuffer.Type dataType) {
        for (MemoryType mt : MemoryType.values()) {
            long bytesFixed = getMemoryBytes(mt, 0, useMode, cacheMode, dataType);
            long bytesPerEx = getMemoryBytes(mt, 1, useMode, cacheMode, dataType) - bytesFixed;

            if(bytesFixed > 0 || bytesPerEx > 0){
                String formatted = String.format("  - %-34s", mt);
                appendFixedPlusVariable(sb, formatted, bytesFixed, bytesPerEx);
            }


        }
    }

    private void appendFixedPlusVariable(StringBuilder sb, String title, long bytesFixed, long bytesPerEx){
        sb.append(title);
        if(bytesFixed > 0){
            sb.append(bytesFixed).append(" bytes");
        }
        if(bytesPerEx > 0){
            if(bytesFixed > 0){
                sb.append(" + ");
            }
            sb.append("nExamples * ").append(bytesPerEx).append(" bytes");
        }
        sb.append("\n");
    }

}
