package org.deeplearning4j.zoo;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * A zoo model is instantiable, returns information about itself, and can download
 * pretrained models if available.
 */
@Slf4j
public abstract class ZooModel<T> implements InstantiableModel {

    public String pretrainedUrl() {
        return null;
    }

    public boolean pretrainedAvailable() {
        if (pretrainedUrl() == null)
            return false;
        else
            return true;
    }

    public Model initPretrained() throws IOException {
        if (pretrainedUrl() == null)
            throw new UnsupportedOperationException("Pretrained weights are not available for this model.");

        String localFilename = new File(pretrainedUrl()).getName();
        File cachedFile = new File(System.getProperty("user.home"), "/.deeplearning4j/" + localFilename);
        cachedFile.mkdirs();

        if(cachedFile.isDirectory())
            cachedFile.delete();

        if (!cachedFile.exists()) {
            log.info("Downloading model to " + cachedFile.toString());
            FileUtils.copyURLToFile(new URL(pretrainedUrl()), cachedFile);
        } else {
            log.info("Using cached model at " + cachedFile.toString());
        }

        if (modelType() == MultiLayerNetwork.class) {
            return ModelSerializer.restoreComputationGraph(cachedFile);
        } else if (modelType() == ComputationGraph.class) {
            return ModelSerializer.restoreComputationGraph(cachedFile);
        } else {
            throw new UnsupportedOperationException(
                            "Pretrained models are only supported for MultiLayerNetwork and ComputationGraph.");
        }
    }
}
