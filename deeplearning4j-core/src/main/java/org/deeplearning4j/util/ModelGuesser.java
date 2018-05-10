package org.deeplearning4j.util;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;

import java.io.*;
import java.util.UUID;

/**
 * Guess a model from the given path
 * @author Adam Gibson
 */
@Slf4j
public class ModelGuesser {


    /**
     * A facade for {@link ModelSerializer#restoreNormalizerFromInputStream(InputStream)}
     * @param is the input stream to load form
     * @return the loaded normalizer
     * @throws IOException
     */
    public static Normalizer<?> loadNormalizer(InputStream is) throws IOException {
        return ModelSerializer.restoreNormalizerFromInputStream(is);
    }

    /**
     * A facade for {@link ModelSerializer#restoreNormalizerFromFile(File)}
     * @param path the path to the file
     * @return the loaded normalizer
     */
    public static Normalizer<?> loadNormalizer(String path) {
        return ModelSerializer.restoreNormalizerFromFile(new File(path));
    }



    /**
     * Load the model from the given file path
     * @param path the path of the file to "guess"
     *
     * @return the loaded model
     * @throws Exception
     */
    public static Object loadConfigGuess(String path) throws Exception {
        String input = FileUtils.readFileToString(new File(path));
        //note here that we load json BEFORE YAML. YAML
        //turns out to load just fine *accidentally*
        try {
            return MultiLayerConfiguration.fromJson(input);
        } catch (Exception e) {
            log.warn("Tried multi layer config from json", e);
            try {
                return KerasModelImport.importKerasModelConfiguration(path);
            } catch (Exception e1) {
                log.warn("Tried keras model config", e);
                try {
                    return KerasModelImport.importKerasSequentialConfiguration(path);
                } catch (Exception e2) {
                    log.warn("Tried keras sequence config", e);
                    try {
                        return ComputationGraphConfiguration.fromJson(input);
                    } catch (Exception e3) {
                        log.warn("Tried computation graph from json");
                        try {
                            return MultiLayerConfiguration.fromYaml(input);
                        } catch (Exception e4) {
                            log.warn("Tried multi layer configuration from yaml");
                            try {
                                return ComputationGraphConfiguration.fromYaml(input);
                            } catch (Exception e5) {
                                throw new ModelGuesserException("Unable to load configuration from path " + path
                                        + " (invalid config file or not a known config type)");
                            }
                        }
                    }
                }
            }
        }
    }


    /**
     * Load the model from the given input stream
     * @param stream the path of the file to "guess"
     *
     * @return the loaded model
     * @throws Exception
     */
    public static Object loadConfigGuess(InputStream stream) throws Exception {
        File tmp = new File("model-" + UUID.randomUUID().toString());
        BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(new FileOutputStream(tmp));
        IOUtils.copy(stream, bufferedOutputStream);
        bufferedOutputStream.flush();
        bufferedOutputStream.close();
        tmp.deleteOnExit();
        Object load = loadConfigGuess(tmp.getAbsolutePath());
        tmp.delete();
        return load;
    }

    /**
     * Load the model from the given file path
     * @param path the path of the file to "guess"
     *
     * @return the loaded model
     * @throws Exception
     */
    public static Model loadModelGuess(String path) throws Exception {
        try {
            return ModelSerializer.restoreMultiLayerNetwork(new File(path), true);
        } catch (Exception e) {
            log.warn("Tried multi layer network");
            try {
                return ModelSerializer.restoreComputationGraph(new File(path), true);
            } catch (Exception e1) {
                log.warn("Tried computation graph");
                try {
                    return ModelSerializer.restoreMultiLayerNetwork(new File(path), false);
                } catch (Exception e4) {
                    try {
                        return ModelSerializer.restoreComputationGraph(new File(path), false);
                    } catch (Exception e5) {
                        try {
                            return KerasModelImport.importKerasModelAndWeights(path);
                        } catch (Exception e2) {
                            log.warn("Tried multi layer network keras");
                            try {
                                return KerasModelImport.importKerasSequentialModelAndWeights(path);

                            } catch (Exception e3) {
                                throw new ModelGuesserException("Unable to load model from path " + path
                                        + " (invalid model file or not a known model type)");
                            }
                        }
                    }

                }



            }
        }
    }


    /**
     * Load the model from the given input stream
     * @param stream the path of the file to "guess"
     *
     * @return the loaded model
     * @throws Exception
     */
    public static Model loadModelGuess(InputStream stream) throws Exception {
        return loadModelGuess(stream, null);
    }

    /**
     * Load the model from the given input stream
     * @param stream the path of the file to "guess"
     * @param directory the directory in which to create any temporary files
     *
     * @return the loaded model
     * @throws Exception
     */
    @Deprecated
    public static Model loadModelGuess(InputStream stream, File tempFileDirectory) throws Exception {
        //Currently (Nov 2017): KerasModelImport doesn't support loading from input streams
        //Simplest solution here: write to a temporary file
        File f = File.createTempFile("loadModelGuess",".bin",tempFileDirectory);
        f.deleteOnExit();

        try (OutputStream os = new BufferedOutputStream(new FileOutputStream(f))) {
            IOUtils.copy(stream, os);
            os.flush();
            return loadModelGuess(f.getAbsolutePath());
        } catch (ModelGuesserException e){
            throw new ModelGuesserException("Unable to load model from input stream (invalid model file not a known model type)");
        } finally {
            f.delete();
        }
    }



}
