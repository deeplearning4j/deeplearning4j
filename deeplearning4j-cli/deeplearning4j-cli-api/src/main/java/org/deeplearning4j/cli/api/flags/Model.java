package org.deeplearning4j.cli.api.flags;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.cli.subcommands.SubCommand;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import java.io.File;

/**
 * Model flag for setting model configurations
 *
 * @author sonali
 */
public class Model implements Flag {
    /**
     * JSON model configuration passed in
     * If you are entering a MultiLayerConfiguration JSON,
     * your file name MUST contain '_multi'.
     * Otherwise, it will be processed as a regular
     * NeuralNetConfiguration
     *
     * Takes in JSON file path
     * Checks file path for indication of MultiLayer
     * Reads JSON file to string
     * Creates neural net configuration from string config
     *
     */
    @Override
    public <E> E value(String value) throws Exception {
        Boolean isMultiLayer = value.contains("_multi");
        String json = FileUtils.readFileToString(new File(value));

        if (isMultiLayer) {
            return (E) MultiLayerConfiguration.fromJson(json);
        }
        else {
            return (E) NeuralNetConfiguration.fromJson(json);
        }
    }

}
