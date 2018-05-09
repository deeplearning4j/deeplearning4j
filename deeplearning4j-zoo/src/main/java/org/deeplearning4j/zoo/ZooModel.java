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
import java.util.zip.Adler32;
import java.util.zip.Checksum;

/**
 * A zoo model is instantiable, returns information about itself, and can download
 * pretrained models if available.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public abstract class ZooModel<T> implements InstantiableModel {

    public static File ROOT_CACHE_DIR = new File(System.getProperty("user.home"), "/.deeplearning4j/");

    public boolean pretrainedAvailable(PretrainedType pretrainedType) {
        if (pretrainedUrl(pretrainedType) == null)
            return false;
        else
            return true;
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
    public <M extends Model> M initPretrained(PretrainedType pretrainedType) throws IOException {
        String remoteUrl = pretrainedUrl(pretrainedType);
        if (remoteUrl == null)
            throw new UnsupportedOperationException(
                            "Pretrained " + pretrainedType + " weights are not available for this model.");

        String localFilename = new File(remoteUrl).getName();

        ROOT_CACHE_DIR.mkdirs();
        File cachedFile = new File(ROOT_CACHE_DIR.getAbsolutePath(), localFilename);

        if (!cachedFile.exists()) {
            log.info("Downloading model to " + cachedFile.toString());
            FileUtils.copyURLToFile(new URL(remoteUrl), cachedFile);
        } else {
            log.info("Using cached model at " + cachedFile.toString());
        }

        long expectedChecksum = pretrainedChecksum(pretrainedType);
        if (expectedChecksum != 0L) {
            log.info("Verifying download...");
            Checksum adler = new Adler32();
            FileUtils.checksum(cachedFile, adler);
            long localChecksum = adler.getValue();
            log.info("Checksum local is " + localChecksum + ", expecting " + expectedChecksum);

            if (expectedChecksum != localChecksum) {
                log.error("Checksums do not match. Cleaning up files and failing...");
                cachedFile.delete();
                throw new IllegalStateException(
                                "Pretrained model file failed checksum. If this error persists, please open an issue at https://github.com/deeplearning4j/deeplearning4j.");
            }
        }

        if (modelType() == MultiLayerNetwork.class) {
            return (M) ModelSerializer.restoreMultiLayerNetwork(cachedFile);
        } else if (modelType() == ComputationGraph.class) {
            return (M) ModelSerializer.restoreComputationGraph(cachedFile);
        } else {
            throw new UnsupportedOperationException(
                            "Pretrained models are only supported for MultiLayerNetwork and ComputationGraph.");
        }
    }
}
