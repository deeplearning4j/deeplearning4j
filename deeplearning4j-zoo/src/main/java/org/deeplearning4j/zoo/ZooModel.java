package org.deeplearning4j.zoo;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * A zoo model is instantiable, returns information about itself, and can download
 * pretrained models if available.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public abstract class ZooModel<T> implements InstantiableModel {

    public String pretrainedImageNetUrl() {
        return null;
    }

    public String pretrainedMnistUrl() {
        return null;
    }

    public boolean pretrainedAvailable(PretrainedType pretrainedType) {
        boolean available;
        switch (pretrainedType) {
            case IMAGENET:
                if (pretrainedImageNetUrl() == null)
                    available = false;
                else
                    available = true;
                break;
            case MNIST:
                if (pretrainedMnistUrl() == null)
                    available = false;
                else
                    available = true;
                break;
            default:
                available = false;
                break;
        }
        return available;
    }

    /**
     * By default, will return a pretrained ImageNet if available.
     *
     * @return
     * @throws IOException
     */
    public Model initPretrained() throws IOException {
        return initPretrained(PretrainedType.IMAGENET);
    }

    /**
     * Returns a pretrained model for the given dataset, if available.
     *
     * @param pretrainedType
     * @return
     * @throws IOException
     */
    public Model initPretrained(PretrainedType pretrainedType) throws IOException {
        String localFilename;
        String remoteUrl;
        switch (pretrainedType) {
            case IMAGENET:
                if (pretrainedImageNetUrl() == null)
                    throw new UnsupportedOperationException(
                                    "Pretrained ImageNet weights are not available for this model.");

                localFilename = new File(pretrainedImageNetUrl()).getName();
                remoteUrl = pretrainedImageNetUrl();
                break;
            case MNIST:
                if (pretrainedMnistUrl() == null)
                    throw new UnsupportedOperationException(
                                    "Pretrained MNIST weights are not available for this model.");

                localFilename = new File(pretrainedMnistUrl()).getName();
                remoteUrl = pretrainedMnistUrl();
                break;
            default:
                throw new UnsupportedOperationException("Only ImageNet and MNIST pretrained models are supported.");
        }

        File cachedFile = new File(System.getProperty("user.home"), "/.deeplearning4j/" + localFilename);
        cachedFile.mkdirs();

        if (cachedFile.isDirectory())
            cachedFile.delete();

        if (!cachedFile.exists()) {
            log.info("Downloading model to " + cachedFile.toString());
            FileUtils.copyURLToFile(new URL(remoteUrl), cachedFile);
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
