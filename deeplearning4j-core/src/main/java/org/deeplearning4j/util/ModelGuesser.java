package org.deeplearning4j.util;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.UUID;

/**
 * Guess a model from the given path
 * @author Adam Gibson
 */
public class ModelGuesser {



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
        }catch (Exception e) {
            try {
                return KerasModelImport.importKerasModelConfiguration(path);
            }catch(Exception e1) {
                try {
                    return KerasModelImport.importKerasSequentialConfiguration(path);
                }catch (Exception e2) {
                    try {
                        return ComputationGraphConfiguration.fromJson(input);
                    }catch(Exception e3) {
                        try {
                            return MultiLayerConfiguration.fromYaml(input);
                        }catch (Exception e4) {
                            try {
                                return ComputationGraphConfiguration.fromYaml(input);
                            }catch(Exception e5) {
                                throw e5;
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
        IOUtils.copy(stream,bufferedOutputStream);
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
            return ModelSerializer.restoreMultiLayerNetwork(new File(path),true);
        }catch (Exception e) {
            try {
                return ModelSerializer.restoreComputationGraph(new File(path),true);
            }catch(Exception e1) {
                try {
                    return KerasModelImport.importKerasModelAndWeights(path);
                }catch(Exception e2) {
                    try {
                        return KerasModelImport.importKerasSequentialModelAndWeights(path);

                    }catch(Exception e3) {
                        throw e3;
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
        try {
            return ModelSerializer.restoreMultiLayerNetwork(stream,true);
        }catch (Exception e) {
            try {
                return ModelSerializer.restoreComputationGraph(stream,true);
            }catch(Exception e1) {
                try {
                    return KerasModelImport.importKerasModelAndWeights(stream);
                }catch(Exception e2) {
                    try {
                        return KerasModelImport.importKerasSequentialModelAndWeights(stream);

                    }catch(Exception e3) {
                        throw e3;
                    }
                }
            }
        }
    }



}
