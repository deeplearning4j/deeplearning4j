package org.deeplearning4j.zoo;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.File;
import java.io.IOException;

/**
 * A zoo model is instantiable, returns information about itself, and can download
 * pretrained models if available.
 */
public abstract class ZooModel<T> implements InstantiableModel {

    public boolean pretrainedAvailable() {
        return false;
    }

    public Model getPretrained() throws IOException {
        String modelFilename = zooType().toString().toLowerCase()+"."+getLatestRevision()+".zip";

        File cachedFile = new File("");
        if(modelType() == MultiLayerNetwork.class) {
            return ModelSerializer.restoreComputationGraph(cachedFile);
        }
        else if(modelType() == ComputationGraph.class) {
            return ModelSerializer.restoreComputationGraph(cachedFile);
        }
        else {
            throw new UnsupportedOperationException("Pretrained models are only supported for MultiLayer and Compgraph.");
        }
    }

    public int getLatestRevision() {
        // Depends on URL scheme not yet implemented
        throw new NotImplementedException();
    }
}
