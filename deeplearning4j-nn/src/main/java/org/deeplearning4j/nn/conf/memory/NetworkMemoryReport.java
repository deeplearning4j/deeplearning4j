package org.deeplearning4j.nn.conf.memory;

import lombok.NonNull;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.buffer.DataBuffer;

import java.util.Map;

/**
 * Created by Alex on 13/07/2017.
 */

public class NetworkMemoryReport extends MemoryReport {

    private final Map<String,LayerMemoryReport> layerAndVertexReports;
    private final Class<? extends Model> modelClass;
    private final String modelName;

    public NetworkMemoryReport(@NonNull Map<String,LayerMemoryReport> layerAndVertexReports,
                               @NonNull Class<? extends Model> modelClass,
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
    public long getMemoryBytes(MemoryType memoryType, int minibatchSize, MemoryUseMode memoryUseMode, DataBuffer.Type dataType) {
        long totalBytes = 0;
        for(LayerMemoryReport lmr : layerAndVertexReports.values()){

            totalBytes += lmr.getMemoryBytes(memoryType, minibatchSize, memoryUseMode, dataType);
        }

        return totalBytes;
    }

    @Override
    public String toString(){
        //TODO
        return "NetworkMemoryReport()";
    }

}
