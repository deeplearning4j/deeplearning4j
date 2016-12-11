package org.deeplearning4j.nn.modelimport.keras;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.hdf5;
import org.deeplearning4j.berkeley.StringUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.bytedeco.javacpp.hdf5.H5F_ACC_RDONLY;
import static org.bytedeco.javacpp.hdf5.H5O_TYPE_DATASET;
import static org.bytedeco.javacpp.hdf5.H5O_TYPE_GROUP;
import static org.deeplearning4j.nn.modelimport.keras.LayerConfiguration.KERAS_LAYER_PROPERTY_DIM_ORDERING;
import static org.deeplearning4j.nn.modelimport.keras.LayerConfiguration.KERAS_DIM_ORDERING_TENSORFLOW;
import static org.deeplearning4j.nn.modelimport.keras.LayerConfiguration.KERAS_DIM_ORDERING_THEANO;
import static org.deeplearning4j.nn.modelimport.keras.ModelConfiguration.*;

/**
 * Routines for importing saved Keras models.
 *
 * @author dave@skymind.io
 */
public class Model {
    private static Logger log = LoggerFactory.getLogger(Model.class);

    private Model() {}

    /**
     * Load Keras model saved using model.save_model(...).
     *
     * @param modelHdf5Filename    path to HDF5 archive storing Keras model
     * @return                     DL4J Model interface
     * @throws IOException
     * @see org.deeplearning4j.nn.api.Model
     */
    public static org.deeplearning4j.nn.api.Model importModel(String modelHdf5Filename) throws IOException {
        /* Read model and training configurations from top-level attributes. */
        hdf5.H5File file = new hdf5.H5File(modelHdf5Filename, H5F_ACC_RDONLY);
        String modelJson = readJsonStringFromHdf5Attribute(file, "model_config");
        String trainingJson = readJsonStringFromHdf5Attribute(file, "training_config");
        file.close();

        /* Import model configuration. */
        org.deeplearning4j.nn.api.Model model = null;
        if (modelIsSequential(modelJson)) {
            MultiLayerConfiguration modelConfig = importSequentialModelConfig(modelJson, trainingJson);
            model = new MultiLayerNetwork(modelConfig);
            ((MultiLayerNetwork)model).init();
        } else {
            ComputationGraphConfiguration modelConfig = importModelConfig(modelJson, trainingJson);
            model = new ComputationGraph(modelConfig);
            ((ComputationGraph)model).init();
        }

        /* Read weights from model archive. */
        Map<String, Object> configMap = getLayerConfigurationAsMap(modelJson);
        Map<String,Map<String,INDArray>> weights = readWeightsFromHdf5(modelHdf5Filename, "/model_weights");

        /* Import weights into model. */
        importWeights(model, weights, configMap);
        return model;
    }

    /**
     * Load Keras model where the config and weights were saved separately using calls to
     * model.to_json() and model.save_weights(...).
     *
     * @param configJsonFilename    path to JSON file storing Keras Functional API model configuration
     * @param weightsHdf5Filename   path to HDF5 archive storing Keras Functional API model weights
     * @return                      DL4J Model interface
     * @throws IOException
     */
    public static org.deeplearning4j.nn.api.Model importModel(String configJsonFilename, String weightsHdf5Filename)
            throws IOException {
        /* Read model configuration from JSON file. */
        String modelJson = new String(Files.readAllBytes(Paths.get(configJsonFilename)));

        /* Import model configuration. */
        org.deeplearning4j.nn.api.Model model = null;
        if (modelIsSequential(modelJson)) {
            MultiLayerConfiguration modelConfig = importSequentialModelConfig(modelJson);
            model = new MultiLayerNetwork(modelConfig);
            ((MultiLayerNetwork)model).init();
        } else {
            ComputationGraphConfiguration modelConfig = importModelConfig(modelJson);
            model = new ComputationGraph(modelConfig);
            ((ComputationGraph)model).init();
        }

        /* Open weights archive and read weights. */
        Map<String,Map<String,INDArray>> weights = readWeightsFromHdf5(weightsHdf5Filename, "/");

        /* Import weights into model. */
        Map<String, Object> configMap = getLayerConfigurationAsMap(modelJson);
        importWeights(model, weights, configMap);
        return model;
    }

    /**
     * Imports a Keras Sequential model saved using model.save_model(...). Model
     * configuration and weights are loaded from single HDF5 archive.
     *
     * @param modelHdf5Filename    path to HDF5 archive storing Keras Sequential model
     * @return                     DL4J MultiLayerNetwork
     * @throws IOException
     * @see MultiLayerNetwork
     * @see MultiLayerConfiguration
     * @deprecated importModel
     */
    @Deprecated
    public static MultiLayerNetwork importSequentialModel(String modelHdf5Filename) throws IOException {
        return (MultiLayerNetwork)importModel(modelHdf5Filename);
    }

    /**
     * Imports a Keras Functional API model saved using model.save_model(...). Model
     * configuration and weights are loaded from single HDF5 archive.
     *
     * @param modelHdf5Filename  path to HDF5 archive storing Keras Functional API model
     * @return                   DL4J ComputationGraph
     * @throws IOException
     * @see    ComputationGraph
     * @see    ComputationGraphConfiguration
     * @deprecated importModel
     */
    @Deprecated
    public static ComputationGraph importFunctionalApiModel(String modelHdf5Filename) throws IOException {
        ComputationGraph model = (ComputationGraph)importModel(modelHdf5Filename);
        return model;
    }

    /**
     * Imports a Keras Sequential model saved using model.save_model(...). Model
     * configuration and weights are loaded from single HDF5 archive.
     *
     * @param configJsonFilename     path to JSON file storing Keras Functional API model configuration
     * @param weightsHdf5Filename    path to HDF5 archive storing Keras Functional API model weights
     * @return                       DL4J MultiLayerNetwork
     * @throws IOException
     * @see MultiLayerNetwork
     * @see MultiLayerConfiguration
     * @deprecated importModel
     */
    @Deprecated
    public static MultiLayerNetwork importSequentialModel(String configJsonFilename, String weightsHdf5Filename)
            throws IOException {
        return (MultiLayerNetwork)importModel(configJsonFilename, weightsHdf5Filename);
    }

    /**
     * Imports a Keras Functional API model saved using model.save_model(...). Model
     * configuration and weights are loaded from single HDF5 archive.
     *
     * @param configJsonFilename     path to JSON file storing Keras Functional API model configuration
     * @param weightsHdf5Filename    path to HDF5 archive storing Keras Functional API model weights
     * @return                       DL4J ComputationGraph
     * @throws IOException
     * @see    ComputationGraph
     * @see    ComputationGraphConfiguration
     * @deprecated importModel
     */
    @Deprecated
    public static ComputationGraph importFunctionalApiModel(String configJsonFilename, String weightsHdf5Filename)
            throws IOException {
        ComputationGraph model = (ComputationGraph)importModel(configJsonFilename, weightsHdf5Filename);
        return model;
    }

    /**
     * Read Keras model weights from HDF5 Group into nested Map of INDArrays, where outer
     * keys are layer names and inner keys are parameter names.
     *
     * @param weightsHdf5Filename    name of HDF5 weights archive file
     * @param weightsGroupName       name of root HDF5 Group storing all Keras weights for single model
     * @return              nested Map from layer names to parameter names to INDArrays
     */
    private static Map<String,Map<String,INDArray>> readWeightsFromHdf5(String weightsHdf5Filename, String weightsGroupName) {
        hdf5.H5File file = new hdf5.H5File(weightsHdf5Filename, H5F_ACC_RDONLY);
        hdf5.Group weightsGroup = file.asCommonFG().openGroup(weightsGroupName);

        Map<String,Map<String,INDArray>> weightsMap = new HashMap<String,Map<String,INDArray>>();

        List<hdf5.Group> groups = new ArrayList<hdf5.Group>();
        groups.add(weightsGroup);
        while (!groups.isEmpty()) {
            hdf5.Group g = groups.remove(0);
            String groupName = g.getObjName().getString();
            for (int i = 0; i < g.asCommonFG().getNumObjs(); i++) {
                BytePointer objPtr = g.asCommonFG().getObjnameByIdx(i);
                String objName = objPtr.getString();
                int objType = g.asCommonFG().childObjType(objPtr);
                switch (objType) {
                    case H5O_TYPE_DATASET:
                        hdf5.DataSet d = g.asCommonFG().openDataSet(objPtr);
                        hdf5.DataSpace space = d.getSpace();
                        int nbDims = (int)space.getSimpleExtentNdims();
                        long[] dims = new long[nbDims];
                        space.getSimpleExtentDims(dims);
                        float[] weightBuffer = null;
                        FloatPointer fp = null;
                        int j = 0;
                        INDArray weights = null;
                        switch (nbDims) {
                            case 4: /* 2D Convolution weights */
                                weightBuffer = new float[(int)(dims[0]*dims[1]*dims[2]*dims[3])];
                                fp = new FloatPointer(weightBuffer);
                                d.read(fp, new hdf5.DataType(hdf5.PredType.NATIVE_FLOAT()));
                                fp.get(weightBuffer);
                                weights = Nd4j.create((int)dims[0], (int)dims[1], (int)dims[2], (int)dims[3]);
                                j = 0;
                                for (int i1 = 0; i1 < dims[0]; i1++)
                                    for (int i2 = 0; i2 < dims[1]; i2++)
                                        for (int i3 = 0; i3 < dims[2]; i3++)
                                            for (int i4 = 0; i4 < dims[3]; i4++)
                                                weights.putScalar(i1, i2, i3, i4, weightBuffer[j++]);
                                break;
                            case 2: /* Dense and Recurrent weights */
                                weightBuffer = new float[(int)(dims[0]*dims[1])];
                                fp = new FloatPointer(weightBuffer);
                                d.read(fp, new hdf5.DataType(hdf5.PredType.NATIVE_FLOAT()));
                                fp.get(weightBuffer);
                                weights = Nd4j.create((int)dims[0], (int)dims[1]);
                                j = 0;
                                for (int i1 = 0; i1 < dims[0]; i1++)
                                    for (int i2 = 0; i2 < dims[1]; i2++)
                                        weights.putScalar(i1, i2, weightBuffer[j++]);
                                break;
                            case 1: /* bias */
                                weightBuffer = new float[(int)dims[0]];
                                fp = new FloatPointer(weightBuffer);
                                d.read(fp, new hdf5.DataType(hdf5.PredType.NATIVE_FLOAT()));
                                fp.get(weightBuffer);
                                weights = Nd4j.create((int)dims[0]);
                                j = 0;
                                for (int i1 = 0; i1 < dims[0]; i1++)
                                    weights.putScalar(i1, weightBuffer[j++]);
                                break;
                            default:
                                throw new IncompatibleKerasConfigurationException("Cannot import weights with rank " + nbDims);

                        }
                        /* Keras parameter names are typically formatted as [layer name]_[layer no]_[parameter]. For
                         * example, the weight matrix in the first Dense layer will be named "dense_1_W."
                         */
                        String[] tokens = objName.split("_");
                        String layerName = StringUtils.join(Arrays.copyOfRange(tokens, 0, tokens.length-1), "_");
                        String paramName = tokens[tokens.length-1];
                        if (!weightsMap.containsKey(layerName))
                            weightsMap.put(layerName, new HashMap<String, INDArray>());
                        weightsMap.get(layerName).put(paramName, weights);
                        d.close();
                        break;
                    case H5O_TYPE_GROUP:
                    default:
                        groups.add(g.asCommonFG().openGroup(objPtr));
                        break;
                }
            }
            g.close();
        }
        file.close();
        return weightsMap;
    }

    /**
     * Helper function to import weights from nested Map into existing model. Depends critically
     * on matched layer and parameter names. In general this seems to be straightforward for most
     * Keras models and layers, but there may be edge cases.
     *
     * @param model             DL4J Model interface
     * @param weights           nested Map from layer names to parameter names to INDArrays
     * @param layerConfigMap    Map from layerName to layerConfig
     * @return                  DL4J Model interface
     * @throws IncompatibleKerasConfigurationException
     */
    private static org.deeplearning4j.nn.api.Model importWeights(org.deeplearning4j.nn.api.Model model,
                                                                 Map<String, Map<String, INDArray>> weights,
                                                                 Map<String, Object> layerConfigMap)
            throws IncompatibleKerasConfigurationException {
        boolean isSequential = (model instanceof MultiLayerNetwork);

        /* TODO: how might this break?
         * - mismatch between layer/parameter names?
         */
        for (String layerName : weights.keySet()) {
            Map<String,Object> layerConfig = (Map<String,Object>)layerConfigMap.get(layerName);
            Layer layer = null;
            if (isSequential)
                layer = ((MultiLayerNetwork)model).getLayer(layerName);
            else
                layer = ((ComputationGraph)model).getLayer(layerName);
            String dimOrdering = null;
            if (layerConfig.containsKey(KERAS_LAYER_PROPERTY_DIM_ORDERING))
                dimOrdering = (String)layerConfig.get(KERAS_LAYER_PROPERTY_DIM_ORDERING);
            for (String kerasParamName : weights.get(layerName).keySet()) {
                String paramName = null;
                /* TensorFlow backend often appends ":" followed by one
                 * or more digits to parameter names, but this is not
                 * reflected in the model config. We must strip it off.
                 */
                Pattern p = Pattern.compile(":\\d+$");
                Matcher m = p.matcher(kerasParamName);
                if (m.find())
                    paramName = m.replaceFirst("");
                else
                    paramName = kerasParamName;
                INDArray W = weights.get(layerName).get(kerasParamName);
                if (layer instanceof ConvolutionLayer && paramName.equals(ConvolutionParamInitializer.WEIGHT_KEY)) {
                    /* Theano and TensorFlow backends store convolutional weights
                     * with a different dimensional ordering than DL4J so we need
                     * to permute them to match.
                     *
                     * DL4J: (# outputs, # channels, # rows, # cols)
                     */
                    if (dimOrdering != null) {
                        if (dimOrdering.equals(KERAS_DIM_ORDERING_TENSORFLOW)) {
                            /* TensorFlow convolutional weights: # rows, # cols, # channels, # outputs */
                            W = W.permute(3, 2, 0, 1);
                        } else if (dimOrdering.equals(KERAS_DIM_ORDERING_THEANO)) {
                            /* Theano convolutional weights: # channels, # rows, # cols, # outputs */
                            W = W.permute(3, 0, 1, 2);
                        } else
                            throw new IncompatibleKerasConfigurationException("Unknown keras backend " + dimOrdering);
                    } else
                        throw new IncompatibleKerasConfigurationException("Convolutional layer must have \"dim_ordering\" property.");
                }
                layer.setParam(paramName, W);
            }
        }
        return model;
    }

    /**
     *
     * @param file
     * @param attribute
     * @return
     */
    private static String readJsonStringFromHdf5Attribute(hdf5.H5File file, String attribute) {
        hdf5.Attribute attr = file.openAttribute(attribute);
        hdf5.VarLenType vl = attr.getVarLenType();
        int bufferSizeMult = 1;
        String jsonString = null;
        /* TODO: find a less hacky way to do this.
         * Reading variable length strings (from attributes) is a giant
         * pain. There does not appear to be any way to determine the
         * length of the string in advance, so we use a hack: choose a
         * buffer size and read the config. If Jackson fails to parse
         * it, then we must not have read the entire config. Increase
         * buffer and repeat.
         */
        while (true) {
            byte[] attrBuffer = new byte[bufferSizeMult * 2000];
            BytePointer attrPointer = new BytePointer(attrBuffer);
            attr.read(vl, attrPointer);
            attrPointer.get(attrBuffer);
            jsonString = new String(attrBuffer);
            ObjectMapper mapper = new ObjectMapper();
            mapper.enable(DeserializationFeature.FAIL_ON_READING_DUP_TREE_KEY);
            try {
                mapper.readTree(jsonString);
                break;
            } catch (IOException e) {}
            bufferSizeMult++;
            if (bufferSizeMult > 100) {
                throw new IncompatibleKerasConfigurationException("Could not read abnormally long Keras config. Please file an issue!");
            }
        }
        return jsonString;
    }
}
