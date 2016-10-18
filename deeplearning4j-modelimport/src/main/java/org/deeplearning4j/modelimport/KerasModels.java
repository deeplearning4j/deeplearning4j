package org.deeplearning4j.modelimport;

import org.apache.commons.lang3.NotImplementedException;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.hdf5.Group;
import org.bytedeco.javacpp.hdf5.DataSet;
import org.bytedeco.javacpp.hdf5.DataSpace;
import org.bytedeco.javacpp.hdf5.DataType;
import org.bytedeco.javacpp.hdf5.PredType;
import org.bytedeco.javacpp.hdf5.H5File;
import org.bytedeco.javacpp.hdf5.Attribute;
import org.bytedeco.javacpp.hdf5.VarLenType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.bytedeco.javacpp.hdf5.H5F_ACC_RDONLY;
import static org.bytedeco.javacpp.hdf5.H5O_TYPE_DATASET;
import static org.bytedeco.javacpp.hdf5.H5O_TYPE_GROUP;

/**
 * Created by davekale on 10/10/16.
 */
public class KerasModels {
    private KerasModels() {}

    public static MultiLayerNetwork importSequentialModel(String modelHdf5Filename) throws IOException {
        H5File file = new H5File();
        file.openFile(modelHdf5Filename, H5F_ACC_RDONLY);
        Attribute attr = file.openAttribute("model_config");
        VarLenType vl = attr.getVarLenType();
        int bufferSizeMult = 1;
        String configJson = null;
        while (true) {
            byte[] attrBuffer = new byte[bufferSizeMult * 500];
            BytePointer attrPointer = new BytePointer(attrBuffer);
            attr.read(vl, attrPointer);
            attrPointer.get(attrBuffer);
            configJson = new String(attrBuffer);
            ObjectMapper mapper = new ObjectMapper();
            mapper.enable(DeserializationFeature.FAIL_ON_READING_DUP_TREE_KEY);
            try {
                mapper.readTree(configJson);
                break;
            } catch (IOException e) {}
            bufferSizeMult++;
        }
        MultiLayerNetwork model = importSequentialModel(configJson, file.asCommonFG().openGroup("model_weights"));
        file.close();
        return model;
    }

    public static MultiLayerNetwork importSequentialModel(String configJsonFilename, String weightsHdf5Filename)
        throws IOException, IncompatibleKerasConfigurationException, NotImplementedException {
        String configJson = readTextFile(configJsonFilename);
        H5File file = new H5File();
        file.openFile(weightsHdf5Filename, H5F_ACC_RDONLY);
        MultiLayerNetwork model = importSequentialModel(configJson, file.asCommonFG().openGroup("/"));
        file.close();
        return model;
    }

    public static MultiLayerConfiguration importSequentialConfig(String jsonConfig)
        throws IncompatibleKerasConfigurationException, NotImplementedException, IOException {
        Map<String,Object> modelConfig = parseJsonString(jsonConfig);
        String arch = (String)modelConfig.get("class_name");
        if (!arch.equals("Sequential"))
            throw new IncompatibleKerasConfigurationException("Found " + arch + " config when trying to load Sequential");

        /* First pass through keras layer configs:
         * - merge Activation and Dropout layers into preceding layers
         *     (TODO: remove future once we add distinct Dropout layers)
         * - determine input shape
         * - identify last layer, convert to output layer
         */
        /* Make a second pass to map keras layers to DL4J layers. */
        int sequenceLength = -1; /* Relevant only for recurrent neural nets. */
        int[] imageSize = new int[]{ -1, -1, -1 }; /* Relevant only for convolutional networks. */
        double dropout = 0.0;
        boolean isInput = true;
        List<Map<String,Object>> layerConfigs = new ArrayList<>();
        Map<String,Object> lastLayer = null;
        for (Object o : (List<Object>)modelConfig.get("config")) {
            String kerasClass = (String)((Map<String,Object>)o).get("class_name");
            Map<String,Object> layerConfig = (Map<String,Object>)((Map<String,Object>)o).get("config");
            layerConfig.put("keras_name", kerasClass);

            /* Most keras layers store output shape in output_dim. */
            int nOut = -1;
            if (layerConfig.containsKey("output_dim"))
                nOut = (Integer)layerConfig.get("output_dim");

            /* Keras input layers are a little complex, but the input shape is consistently
             * stored in the "batch_input_shape" field in the first layer.
             */
            int nIn = -1;
            List<Integer> batchInputShape = null;
            if (layerConfig.containsKey("batch_input_shape")) {
                if (isInput) {
                    batchInputShape = (List<Integer>) layerConfig.get("batch_input_shape");
                    nIn = batchInputShape.get(1);
                } else /* Non-input layers should NOT define batch_input_shape. */
                    throw new IncompatibleKerasConfigurationException("Non-input layer should not specify batch_input_shape");
            } else if (isInput) /* Input layers MUST define batch_input_shape. */
                throw new IncompatibleKerasConfigurationException("Input layer must specify batch_input_shape");

            String dl4jClass = null;
            switch (kerasClass) {
                case "Activation":
                    /* Merge activation into previous layer
                     * TODO: change this -- we actually DO support plain Activation layers
                     */
                    if (!layerConfigs.isEmpty())
                        layerConfigs.get(layerConfigs.size() - 1).put("activation", layerConfig.get("activation"));
                    else
                        throw new IncompatibleKerasConfigurationException("Plain activation layers not supported!");
                    continue;
                case "Dropout":
                    /* Merge dropout into subsequent layer
                     * TODO: change this -- we actually DO support plain Activation layers
                     */
                    dropout = (Double)layerConfig.get("p");
                    continue;
                case "Dense":
                    dl4jClass = "DenseLayer";
                    lastLayer = layerConfig;
                    break;
                case "TimeDistributedDense":
                    dl4jClass = "";
                    lastLayer = layerConfig;
                    break;
                case "LSTM":
                    dl4jClass = "GravesLSTM";
                    if (batchInputShape != null) {
                        nIn = batchInputShape.get(2);
                        sequenceLength = batchInputShape.get(1);
                    }
                    if (!layerConfig.get("activation").equals(layerConfig.get("inner_activation")))
                        throw new IncompatibleKerasConfigurationException("Specifying different activation for inner cells not supported.");
                    if (!layerConfig.get("init").equals(layerConfig.get("inner_init")))
                        System.err.println("Specifying different initialization for inner cells not supported.");
                    if ((Float)layerConfig.get("dropout_U") > 0f)
                        throw new IncompatibleKerasConfigurationException("Recurrent dropout of " + (float) layerConfig.get("dropout_U") + " not supported.");
                    layerConfig.put("dropout", layerConfig.get("dropout_W"));
                    lastLayer = layerConfig;
                    break;
                case "Convolution2D":
                    dl4jClass = "ConvolutionLayer";
                    // TODO: do we need to do anything else with dim ordering?
                    String convDimOrder = (String)layerConfig.get("dim_ordering");
                    switch (convDimOrder) {
                        case "tf":
                            if (batchInputShape != null) {
                                imageSize[0] = batchInputShape.get(1);
                                imageSize[1] = batchInputShape.get(2);
                                imageSize[2] = nIn = batchInputShape.get(3);
                            }
                            break;
                        case "th":
                            if (batchInputShape != null) {
                                imageSize[0] = batchInputShape.get(2);
                                imageSize[1] = batchInputShape.get(3);
                                imageSize[2] = nIn = batchInputShape.get(1);
                            }
                            break;
                        default:
                            throw new IncompatibleKerasConfigurationException("Unknown keras dim ordering in convolutional layer: " + convDimOrder);
                    }
                    nOut = (Integer)layerConfig.get("nb_filter");
                    lastLayer = layerConfig;
                    break;
                case "MaxPooling2D":
                    dl4jClass = "SubsamplingLayer";
                    // TODO: do we need to do anything else with dim ordering?
                    String poolDimOrder = (String)layerConfig.get("dim_ordering");
                    switch (poolDimOrder) {
                        case "tf":
                            if (batchInputShape != null) {
                                nIn = batchInputShape.get(3);
                                imageSize[0] = batchInputShape.get(1);
                                imageSize[1] = batchInputShape.get(2);
                            }
                            break;
                        case "th":
                            if (batchInputShape != null) {
                                nIn = batchInputShape.get(1);
                                imageSize[0] = batchInputShape.get(2);
                                imageSize[1] = batchInputShape.get(3);
                            }
                            break;
                        default:
                            throw new IncompatibleKerasConfigurationException("Unknown keras dim ordering in convolutional layer: " + poolDimOrder);
                    }
                    lastLayer = layerConfig;
                    break;
                case "Flatten":
                    System.err.println("DL4J adds reshaping layers during model compilation");
                    continue;
                default:
                    throw new IncompatibleKerasConfigurationException("Unsupported keras layer type " + kerasClass);
            }
            layerConfig.put("dl4j_name", dl4jClass);
            layerConfig.put("nOut", nOut);
            layerConfig.put("nIn", nIn);

            /* Merge dropout from previous layer.
             * TODO: remove once Dropout layer added to DL4J.
             */
            if (dropout > 0) {
                double oldDropout = layerConfig.containsKey("dropout") ? (Double)layerConfig.get("dropout") : 0.0;
                double newDropout = 1.0 - (1.0 - dropout) * (1.0 - oldDropout);
                layerConfig.put("dropout", newDropout);
                if (oldDropout > 0)
                    System.err.println("Changed layer-defined dropout " + oldDropout + " to " + newDropout +
                                       " because of previous Dropout=" + dropout + " layer");
            }
            dropout = 0;
            isInput = false;
            layerConfigs.add(layerConfig);
        }
        String kerasClass = (String)lastLayer.get("keras_name");
        switch (kerasClass) {
            case "Dense":
                lastLayer.put("dl4j_name", "OutputLayer");
                break;
            case "TimeDistributedDense":
                lastLayer.put("dl4j_name", "RnnOutputLayer");
                break;
            default:
                throw new IncompatibleKerasConfigurationException("Incompatible keras output layer " + kerasClass);
        }
        for (Map<String,Object> layerConfig : layerConfigs)
            if (layerConfig.get("dl4j_name").equals(""))
                throw new IncompatibleKerasConfigurationException("Unsupported keras layer type " + layerConfig.get("dl4j_name"));

        /* Now make second pass through preprocessed layer configs
         * and add them to NeuralNetConfiguration builder using
         * the Builder interface.
         */
        int layerIndex = 0;
        NeuralNetConfiguration.Builder modelBuilder = new NeuralNetConfiguration.Builder();
        NeuralNetConfiguration.ListBuilder listBuilder = modelBuilder.list();
        for (Map<String,Object> layerConfig : layerConfigs) {
            String dl4jClass = (String)layerConfig.get("dl4j_name");

            /* Common layer properties. */
            String activation = layerConfig.containsKey("activation") ? (String)layerConfig.get("activation") : null;
            if (activation != null && activation.equals("linear"))
                activation = "identity";
            String init = layerConfig.containsKey("init") ? (String)layerConfig.get("init") : null;
            dropout = layerConfig.containsKey("dropout") ? (Double)layerConfig.get("dropout") : 0;
            List<Integer> inputShape = null;
            int nIn = (Integer)layerConfig.get("nIn");
            int nOut = (Integer)layerConfig.get("nOut");
            /* Common convolution and pooling layer properties. */
            List<Integer> stride = null;

            Layer.Builder builder = null;
            /* Do layer type-specific stuff here. */
            switch (dl4jClass) {
                case "OutputLayer":
                    builder = new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT);
                    break;
                case "RnnOutputLayer":
                    builder = new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT);
                    break;
                case "DenseLayer":
                    builder = new DenseLayer.Builder();
                    break;
                case "ConvolutionLayer":
                    stride = (List<Integer>)layerConfig.get("subsample");
                    int[] kernelSize = new int[]{ (Integer)layerConfig.get("nb_row"),
                                                  (Integer)layerConfig.get("nb_col") };
                    builder = new ConvolutionLayer.Builder(kernelSize)
                            .stride(stride.get(0), stride.get(1));
                    if (imageSize[0] < 0 || imageSize[1] < 0)
                        throw new IncompatibleKerasConfigurationException("WARNING: input image size must be specified for convolutions!");
                    break;
                case "SubsamplingLayer":
                    stride = (List<Integer>)layerConfig.get("strides");
                    List<Integer> pool = (List<Integer>)layerConfig.get("pool_size");
                    builder = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(pool.get(0), pool.get(1))
                            .stride(stride.get(0), stride.get(1));
                    break;
                case "GravesLSTM":
                    dropout = (Double)layerConfig.get("dropout_W");
                    builder = new GravesLSTM.Builder();
                    String forgetBiasInit = (String)layerConfig.get("forget_bias_init");
                    switch (forgetBiasInit) {
                        case "zero":
                            ((GravesLSTM.Builder)builder).forgetGateBiasInit(0.0);
                            break;
                        case "one":
                            ((GravesLSTM.Builder)builder).forgetGateBiasInit(1.0);
                            break;
                        default:
                            System.err.println("Unsupported bias initialization: " + forgetBiasInit + ".");
                            break;
                    }
                    // TODO: should we print a warning if unroll is false?
                    if (sequenceLength <= 0)
                        System.err.println("WARNING: input sequence length must be specified for truncated BPTT!");
                    // TODO: do we need to do anything with return_sequences?
                    break;
                default:
                    throw new IncompatibleKerasConfigurationException("Unknown layer type " + dl4jClass);
            }

            /* Do generic (common across all layer types) stuff here. */
            builder.name((String)layerConfig.get("name")).activation(activation);
            if (dropout > 0) builder.dropOut(dropout);
            if (nIn > 0) ((FeedForwardLayer.Builder)builder).nIn(nIn);
            if (nOut > 0) ((FeedForwardLayer.Builder)builder).nOut(nOut);

            /* WEIGHT INITIALIZATION
             * TODO: finish mapping keras-to-dl4j weight distributions.
             * Low priority since our focus is on loading trained models.
             *
             * Remaining dl4j distributions: DISTRIBUTION, SIZE, NORMALIZED,
             * VI, RELU, XAVIER
             */
            if (init != null)
                switch (init) {
                    case "uniform":
                        builder.weightInit(WeightInit.UNIFORM);
                        break;
                    case "zero":
                        builder.weightInit(WeightInit.ZERO);
                        builder.biasInit(0.0);
                        break;
                    case "lecun_uniform":
                    case "normal":
                    case "identity":
                    case "orthogonal":
                    case "glorot_normal":
                    case "glorot_uniform":
                    case "he_normal":
                    case "he_uniform":
                    default:
                        System.err.println("Unknown keras weight distribution " + init);
                        builder.weightInit(WeightInit.XAVIER);
                        break;
                }

            /* REGULARIZATION */
            Map<String,Object> W_reg = (Map<String,Object>)layerConfig.get("W_regularizer");
            if (W_reg != null) {
                for (String k : W_reg.keySet()) {
                    switch (k) {
                        case "l1":
                            double l1 = (Double)W_reg.get(k);
                            if (l1 > 0) {
                                modelBuilder.setUseRegularization(true);
                                builder.l1(l1);
                            }
                            break;
                        case "l2":
                            double l2 = (Double)W_reg.get(k);
                            if (l2 > 0) {
                                modelBuilder.setUseRegularization(true);
                                builder.l2(l2);
                            }
                            break;
                        case "name":
                            break;
                        default:
                            throw new IncompatibleKerasConfigurationException("Unknown regularization field: " + k);
                    }
                }
            }
            if (layerConfig.get("b_regularizer") != null)
                throw new NotImplementedException("Bias regularization not implemented");

            //TODO: add exceptions for other unsupported things
            listBuilder.layer(layerIndex++, builder.build());
        }
        listBuilder.backprop(true);

        /* Set Truncated BPTT if Keras model set a fixed sequence length. */
        if (sequenceLength > 0)
            listBuilder.tBPTTBackwardLength(sequenceLength).tBPTTForwardLength(sequenceLength);

        /* Set InputType to convolutional if necessary. */
        if (imageSize[0] > 0 && imageSize[1] > 0)
            listBuilder.setInputType(new InputType.InputTypeConvolutional(imageSize[0], imageSize[1], imageSize[2]));

        return listBuilder.build();
    }

    private static Map<String,Object> parseJsonString(String json) throws IOException {
        /* Parse config JSON string into nested Map<String,Object>. */
        ObjectMapper mapper = new ObjectMapper();
        TypeReference<HashMap<String,Object>> typeRef = new TypeReference<HashMap<String,Object>>() {};
        return mapper.readValue(json, typeRef);
    }

    private static String readTextFile(String filename) throws IOException {
        return new String(Files.readAllBytes(Paths.get(filename)));
    }

    private static MultiLayerNetwork importSequentialModel(String configJson, Group weightsGroup)
        throws IOException {
        MultiLayerConfiguration config = importSequentialConfig(configJson);
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        Map<String,Object> weightsMetadata = extractWeightsMetadataFromConfig(configJson);
        Map<String,Map<String,INDArray>> weights = readWeightsFromHdf5(weightsGroup);
        importWeights(model, weights, weightsMetadata);
        return model;
    }

    private static Map<String, Object> extractWeightsMetadataFromConfig(String configJson) throws IOException {
        Map<String,Object> weightsMetadata = new HashMap<>();
        Map<String,Object> kerasConfig = parseJsonString(configJson);
        List<Map<String,Object>> layers = (List<Map<String,Object>>)kerasConfig.get("config");
        for (Map<String,Object> layer : layers) {
            Map<String,Object> layerConfig = (Map<String,Object>)layer.get("config");
            if (layerConfig.containsKey("dim_ordering") && !weightsMetadata.containsKey("keras_backend"))
                weightsMetadata.put("keras_backend", layerConfig.get("dim_ordering"));
        }
        return weightsMetadata;
    }

    private static Map<String,Map<String,INDArray>> readWeightsFromHdf5(Group weightsGroup) {
        Map<String,Map<String,INDArray>> weightsMap = new HashMap<String,Map<String,INDArray>>();

        List<Group> groups = new ArrayList<Group>();
        groups.add(weightsGroup);
        while (!groups.isEmpty()) {
            Group g = groups.remove(0);
            String groupName = g.getObjName().getString();
            for (int i = 0; i < g.asCommonFG().getNumObjs(); i++) {
                BytePointer objPtr = g.asCommonFG().getObjnameByIdx(i);
                String objName = objPtr.getString();
                int objType = g.asCommonFG().childObjType(objPtr);
                switch (objType) {
                    case H5O_TYPE_DATASET:
                        DataSet d = g.asCommonFG().openDataSet(objPtr);
                        DataSpace space = d.getSpace();
                        int nbDims = (int)space.getSimpleExtentNdims();
                        long[] dims = new long[nbDims];
                        space.getSimpleExtentDims(dims);
                        float[] weightBuffer = null;
                        FloatPointer fp = null;
                        int j = 0;
                        INDArray weights = null;
                        switch (nbDims) {
                            case 4:
                                weightBuffer = new float[(int)(dims[0]*dims[1]*dims[2])];
                                fp = new FloatPointer(weightBuffer);
                                d.read(fp, new DataType(PredType.NATIVE_FLOAT()));
                                fp.get(weightBuffer);
                                weights = Nd4j.create((int)dims[0], (int)dims[1], (int)dims[2], (int)dims[2]);
                                j = 0;
                                for (int i1 = 0; i1 < dims[0]; i1++)
                                    for (int i2 = 0; i2 < dims[1]; i2++)
                                        for (int i3 = 0; i3 < dims[2]; i3++)
                                            for (int i4 = 0; i4 < dims[3]; i4++)
                                                weights.putScalar(i1, i2, i3, i4, weightBuffer[j++]);
                                break;
                            case 2:
                                weightBuffer = new float[(int)(dims[0]*dims[1])];
                                fp = new FloatPointer(weightBuffer);
                                d.read(fp, new DataType(PredType.NATIVE_FLOAT()));
                                fp.get(weightBuffer);
                                weights = Nd4j.create((int)dims[0], (int)dims[1]);
                                j = 0;
                                for (int i1 = 0; i1 < dims[0]; i1++)
                                    for (int i2 = 0; i2 < dims[1]; i2++)
                                        weights.putScalar(i1, i2, weightBuffer[j++]);
                                break;
                            case 1:
                                weightBuffer = new float[(int)dims[0]];
                                fp = new FloatPointer(weightBuffer);
                                d.read(fp, new DataType(PredType.NATIVE_FLOAT()));
                                fp.get(weightBuffer);
                                weights = Nd4j.create((int)dims[0]);
                                j = 0;
                                for (int i1 = 0; i1 < dims[0]; i1++)
                                    weights.putScalar(i1, weightBuffer[j++]);
                                break;
                            default:
                                throw new IncompatibleKerasConfigurationException("Cannot import weights with rank " + nbDims);

                        }
//                        weightsMap.put(objName, weights);
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
        return weightsMap;
    }

    private static MultiLayerNetwork importWeights(MultiLayerNetwork model, Map<String, Map<String, INDArray>> weights,
                                                   Map<String, Object> weightsMetadata)
        throws IncompatibleKerasConfigurationException {
        /* TODO: how might this break?
         * - mismatch between layer/parameter names?
         */
        for (String layerName : weights.keySet())
            for (String paramName : model.getLayer(layerName).paramTable().keySet()) {
                INDArray w = weights.get(layerName).get(paramName);
                org.deeplearning4j.nn.api.Layer layer = model.getLayer(layerName);
                if (layer instanceof org.deeplearning4j.nn.layers.convolution.ConvolutionLayer){
                    /* TODO: swap dimensions if necessary
                     * - theano: (# inputs, # rows, # cols, # outputs)
                     * - tensorflow: (# rows, # cols, # inputs, # outputs)
                     * - dl4j: ?
                     */
                    String kerasBackend = weightsMetadata.containsKey("keras_backend") ?
                        (String)weightsMetadata.get("keras_backend") : "none";
                    switch (kerasBackend) {
                        case "th":
                            break;
                        case "tf":
                            break;
                        default:
                            throw new IncompatibleKerasConfigurationException("Unknown keras backend " + kerasBackend);
                    }
                } else if (layer instanceof org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer){
                    /* TODO: swap dimensions if necessary
                     * - keras: (# input, # output)
                     * - dl4j: ?
                     */
                }
                model.getLayer(layerName).setParam(paramName, w);
            }
        return model;
    }
}
