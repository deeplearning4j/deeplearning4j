package org.deeplearning4j.cli.api.flags;

import org.deeplearning4j.cli.subcommands.SubCommand;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

/**
 * Model flag for setting model configurations
 *
 * @author sonali
 */
public class Model implements Flag {
    /**
     * JSON model configuration passed in
     * If you are entering a MultiLayerConfiguration JSON,
     * your file name MUST end with '_multi.json'.
     * Otherwise, it will be processed as a regular
     * NeuralNetConfiguration
     *
     */
    @Override
    public <E> E value(String value) throws Exception {
        //take in JSON
        //figure out if it's multilayer or normal
        //return a configuration

        Boolean isMultiLayer = value.contains("_multi");

        if (isMultiLayer) {
            return (E) MultiLayerConfiguration.fromJson(value);
        }
        else {
            return (E) NeuralNetConfiguration.fromJson(value);
        }
    }

}
