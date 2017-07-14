package org.deeplearning4j.nn.conf.memory;

import lombok.NonNull;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.CacheMode;
import org.nd4j.linalg.api.buffer.DataBuffer;

import java.util.Map;

/**
 * Created by Alex on 13/07/2017.
 */

public class NetworkMemoryReport extends MemoryReport {

    private final Map<String,MemoryReport> layerAndVertexReports;
    private final Class<?> modelClass;
    private final String modelName;

    public NetworkMemoryReport(@NonNull Map<String,MemoryReport> layerAndVertexReports,
                               @NonNull Class<?> modelClass,
                               String modelName){
        this.layerAndVertexReports = layerAndVertexReports;
        this.modelClass = modelClass;
        this.modelName = modelName;
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
        for(MemoryReport lmr : layerAndVertexReports.values()){

            totalBytes += lmr.getMemoryBytes(memoryType, minibatchSize, memoryUseMode, cacheMode, dataType);
        }

        return totalBytes;
    }

    @Override
    public String toString(){
        //TODO
        return "NetworkMemoryReport()";
    }

}
