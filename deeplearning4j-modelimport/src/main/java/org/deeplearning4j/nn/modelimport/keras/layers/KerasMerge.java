package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;

import java.util.Map;

/**
 * Imports a Keras Merge layer as a DL4J Merge (graph) vertex.
 *
 * TODO: handle axes arguments that alter merge behavior (requires changes to DL4J?)
 * TODO: unsupported merge modes (require changes to DL4J)
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasMerge extends KerasLayer {

    public static final String LAYER_FIELD_MODE = "mode";
    public static final String LAYER_MERGE_MODE_SUM = "sum";
    public static final String LAYER_MERGE_MODE_MUL = "mul";
    public static final String LAYER_MERGE_MODE_CONCAT = "concat";
    public static final String LAYER_MERGE_MODE_AVE = "ave";
    public static final String LAYER_MERGE_MODE_COS = "cos";
    public static final String LAYER_MERGE_MODE_DOT = "dot";
    public static final String LAYER_MERGE_MODE_MAX = "max";

    private ElementWiseVertex.Op mergeMode = null;

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig   dictionary containing Keras layer configuration.
     *
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasMerge(Map<String, Object> layerConfig)
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
    public KerasMerge(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        this.mergeMode = getMergeMode(layerConfig);
        if (this.mergeMode == null)
            this.vertex = new MergeVertex();
        else
            this.vertex = new ElementWiseVertex(mergeMode);
    }

    public ElementWiseVertex.Op getMergeMode(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_MODE))
            throw new InvalidKerasConfigurationException(
                            "Keras Merge layer config missing " + LAYER_FIELD_MODE + " field");
        ElementWiseVertex.Op op = null;
        String mergeMode = (String) innerConfig.get(LAYER_FIELD_MODE);
        switch (mergeMode) {
            case LAYER_MERGE_MODE_SUM:
                op = ElementWiseVertex.Op.Add;
                break;
            case LAYER_MERGE_MODE_MUL:
                op = ElementWiseVertex.Op.Product;
                break;
            case LAYER_MERGE_MODE_CONCAT:
                // leave null
                break;
            case LAYER_MERGE_MODE_AVE:
            case LAYER_MERGE_MODE_COS:
            case LAYER_MERGE_MODE_DOT:
            case LAYER_MERGE_MODE_MAX:
            default:
                throw new UnsupportedKerasConfigurationException(
                                "Keras Merge layer mode " + mergeMode + " not supported");
        }
        return op;
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
        return this.vertex.getOutputType(-1, inputType);
    }
}
