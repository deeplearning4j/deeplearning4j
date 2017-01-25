package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;

import java.util.Map;

/**
 * Imports a Keras Merge layer as a DL4J Merge (graph) vertex.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasMerge extends KerasLayer {

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig   dictionary containing Keras layer configuration.
     *
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasMerge(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig               dictionary containing Keras layer configuration
     * @param enforceTrainingConfig     whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasMerge(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        this.vertex = new MergeVertex();
    }

    /**
     * Get DL4J MergeVertex.
     *
     * @return  MergeVertex
     */
    public MergeVertex getMergeVertex() {
        return (MergeVertex)this.vertex;
    }

    /**
     * Get layer output type.
     *
     * @param  inputType    Array of InputTypes
     * @return              output type as InputType
     * @throws InvalidKerasConfigurationException
     */
    @Override
    public InputType getOutputType(InputType... inputType) {
        return getMergeVertex().getOutputType(-1, inputType);
    }
}
