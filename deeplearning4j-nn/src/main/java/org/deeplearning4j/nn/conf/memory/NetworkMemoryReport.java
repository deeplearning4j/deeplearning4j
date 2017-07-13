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

    public NetworkMemoryReport(@NonNull Map<String,LayerMemoryReport> layerAndVertexReports, Class<? extends Model> modelClass ){
        this.layerAndVertexReports = layerAndVertexReports;
        this.modelClass = modelClass;
    }

    @Override
    public long getTotalMemoryBytes(int minibatchSize, @NonNull DataBuffer.Type dataType ){
        int bytesPerElem;
        switch (dataType){
            case DOUBLE:
                bytesPerElem = 8;
                break;
            case FLOAT:
                bytesPerElem = 4;
                break;
            case HALF:
                bytesPerElem = 2;
                break;
            default:
                throw new UnsupportedOperationException("Data type not supported: " + dataType);
        }
        long sumBytes = 0;
        for(LayerMemoryReport lmr : layerAndVertexReports.values() ){

        }
    }


    @Override
    public long getMemoryBytes(MemoryType memoryType, int minibatchSize, DataBuffer.Type dataType ){

    }


    @Override
    public String toString(){

    }

}
