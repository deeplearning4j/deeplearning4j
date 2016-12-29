/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.hdf5;
import org.deeplearning4j.berkeley.StringUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.*;
import java.lang.Exception;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.bytedeco.javacpp.hdf5.*;
import static org.deeplearning4j.nn.modelimport.keras.KerasModel.MODEL_CLASS_NAME_MODEL;
import static org.deeplearning4j.nn.modelimport.keras.KerasModel.MODEL_CLASS_NAME_SEQUENTIAL;
import static org.deeplearning4j.nn.modelimport.keras.KerasModel.MODEL_FIELD_CLASS_NAME;

/**
 * Reads stored Keras configurations and weights from one of two archives:
 * either (1) a single HDF5 file storing model and training JSON configurations
 * and weights or (2) separate text file storing model JSON configuration and
 * HDF5 file storing weights.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasModelImport {
    static {
        try {
            /* This is necessary for the call to the BytePointer constructor below. */
            Loader.load(hdf5.class);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private String modelJson;      // model configuration JSON string
    private String trainingJson;   // training configuration JSON string
    private String modelClassName; // Keras model class name
    private Map<String,Map<String,INDArray>> weights; // map from layer to parameter to weights

    /**
     * Load Keras (Functional API) Model saved using model.save_model(...).
     *
     * @param modelHdf5Stream      InputStream containing HDF5 archive storing Keras Model
     * @return                     ComputationGraph
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     * @see ComputationGraph
     */
    public static ComputationGraph importKerasModelAndWeights(InputStream modelHdf5Stream)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasModelImport archive = new KerasModelImport(modelHdf5Stream);
        if (!archive.getModelClassName().equals(MODEL_CLASS_NAME_MODEL))
            throw new InvalidKerasConfigurationException("Expected Keras model class name Model (found " + archive.getModelClassName() + ")");
        KerasModel kerasModel = new KerasModel.ModelBuilder()
                .modelJson(archive.getModelJson())
                .trainingJson(archive.getTrainingJson())
                .weights(archive.getWeights())
                .train(false)
                .buildModel();
        ComputationGraph model = kerasModel.getComputationGraph();
        return model;
    }

    /**
     * Load Keras Sequential model saved using model.save_model(...).
     *
     * @param modelHdf5Stream      InputStream containing HDF5 archive storing Keras Sequential model
     * @return                     ComputationGraph
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     * @see ComputationGraph
     */
    public static MultiLayerNetwork importKerasSequentialModelAndWeights(InputStream modelHdf5Stream)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasModelImport archive = new KerasModelImport(modelHdf5Stream);
        if (!archive.getModelClassName().equals(MODEL_CLASS_NAME_MODEL))
            throw new InvalidKerasConfigurationException("Expected Keras model class name Model (found " + archive.getModelClassName() + ")");
        KerasSequentialModel kerasModel = new KerasModel.ModelBuilder()
                .modelJson(archive.getModelJson())
                .trainingJson(archive.getTrainingJson())
                .weights(archive.getWeights())
                .train(false)
                .buildSequential();
        MultiLayerNetwork model = kerasModel.getMultiLayerNetwork();
        return model;
    }

    /**
     * Load Keras (Functional API) Model saved using model.save_model(...).
     *
     * @param modelHdf5Filename    path to HDF5 archive storing Keras Model
     * @return                     ComputationGraph
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     * @see ComputationGraph
     */
    public static ComputationGraph importKerasModelAndWeights(String modelHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasModelImport archive = new KerasModelImport(modelHdf5Filename);
        if (!archive.getModelClassName().equals(MODEL_CLASS_NAME_MODEL))
            throw new InvalidKerasConfigurationException("Expected Keras model class name Model (found " + archive.getModelClassName() + ")");
        KerasModel kerasModel = new KerasModel.ModelBuilder()
                                        .modelJson(archive.getModelJson())
                                        .trainingJson(archive.getTrainingJson())
                                        .weights(archive.getWeights())
                                        .train(false)
                                        .buildModel();
        ComputationGraph model = kerasModel.getComputationGraph();
        return model;
    }

    /**
     * Load Keras Sequential model saved using model.save_model(...).
     *
     * @param modelHdf5Filename    path to HDF5 archive storing Keras Sequential model
     * @return                     MultiLayerNetwork
     * @throws IOException
     * @see MultiLayerNetwork
     */
    public static MultiLayerNetwork importKerasSequentialModelAndWeights(String modelHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasModelImport archive = new KerasModelImport(modelHdf5Filename);
        if (!archive.getModelClassName().equals(MODEL_CLASS_NAME_SEQUENTIAL))
            throw new InvalidKerasConfigurationException("Expected Keras model class name Sequential (found " + archive.getModelClassName() + ")");
        KerasSequentialModel kerasModel = new KerasSequentialModel.ModelBuilder()
                .modelJson(archive.getModelJson())
                .trainingJson(archive.getTrainingJson())
                .weights(archive.getWeights())
                .train(false)
                .buildSequential();
        MultiLayerNetwork model = kerasModel.getMultiLayerNetwork();
        return model;
    }

    /**
     * Load Keras (Functional API) Model for which the configuration and weights were
     * saved separately using calls to model.to_json() and model.save_weights(...).
     *
     * @param modelJsonFilename     path to JSON file storing Keras Model configuration
     * @param weightsHdf5Filename   path to HDF5 archive storing Keras model weights
     * @return                      ComputationGraph
     * @throws IOException
     * @see ComputationGraph
     */
    public static ComputationGraph importKerasModelAndWeights(String modelJsonFilename, String weightsHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasModelImport archive = new KerasModelImport(modelJsonFilename, weightsHdf5Filename);
        if (!archive.getModelClassName().equals(MODEL_CLASS_NAME_SEQUENTIAL))
            throw new InvalidKerasConfigurationException("Expected Keras model class name Model (found " + archive.getModelClassName() + ")");
        KerasModel kerasModel = new KerasModel.ModelBuilder()
                .modelJson(archive.getModelJson())
                .weights(archive.getWeights())
                .train(false)
                .buildModel();
        ComputationGraph model = kerasModel.getComputationGraph();
        return model;
    }

    /**
     * Load Keras Sequential model for which the configuration and weights were
     * saved separately using calls to model.to_json() and model.save_weights(...).
     *
     * @param modelJsonFilename      path to JSON file storing Keras Sequential model configuration
     * @param weightsHdf5Filename    path to HDF5 archive storing Keras model weights
     * @return                       MultiLayerNetwork
     * @throws IOException
     * @see MultiLayerNetwork
     */
    public static MultiLayerNetwork importKerasSequentialModelAndWeights(String modelJsonFilename, String weightsHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasModelImport archive = new KerasModelImport(modelJsonFilename, weightsHdf5Filename);
        if (!archive.getModelClassName().equals(MODEL_CLASS_NAME_SEQUENTIAL))
            throw new InvalidKerasConfigurationException("Expected Keras model class name Sequential (found " + archive.getModelClassName() + ")");
        KerasSequentialModel kerasModel = new KerasSequentialModel.ModelBuilder()
                .modelJson(archive.getModelJson())
                .trainingJson(archive.getTrainingJson())
                .weights(archive.getWeights())
                .train(false)
                .buildSequential();
        MultiLayerNetwork model = kerasModel.getMultiLayerNetwork();
        return model;
    }

    /**
     * Load Keras (Functional API) Model for which the configuration was saved
     * separately using calls to model.to_json() and model.save_weights(...).
     *
     * @param modelJsonFilename      path to JSON file storing Keras Model configuration
     * @return                       ComputationGraph
     * @throws IOException
     * @see ComputationGraph
     */
    public static ComputationGraphConfiguration importKerasModelConfiguration(String modelJsonFilename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        String modelJson = new String(Files.readAllBytes(Paths.get(modelJsonFilename)));
        KerasModel kerasModel = new KerasModel.ModelBuilder()
                .modelJson(modelJson)
                .train(false)
                .buildModel();
        return kerasModel.getComputationGraphConfiguration();
    }

    /**
     * Load Keras Sequential model for which the configuration was saved
     * separately using calls to model.to_json() and model.save_weights(...).
     *
     * @param modelJsonFilename      path to JSON file storing Keras Sequential model configuration
     * @return                       MultiLayerNetwork
     * @throws IOException
     * @see MultiLayerNetwork
     */
    public static MultiLayerConfiguration importKerasSequentialConfiguration(String modelJsonFilename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        String modelJson = new String(Files.readAllBytes(Paths.get(modelJsonFilename)));
        KerasSequentialModel kerasModel = new KerasSequentialModel.ModelBuilder()
                .modelJson(modelJson)
                .train(false)
                .buildSequential();
        return kerasModel.getMultiLayerConfiguration();
    }

    /**
     * Constructor from HDF5 model archive stored in InputStream.
     *
     * @param modelHdf5Stream     InputStream containing HDF5 archive of Keras model
     * @throws IOException
     *
     * TODO: Currently, this constructor does not work. It does not appear to be
     * possible to open an HDF5 archive from raw bytes.
     */
    public KerasModelImport(InputStream modelHdf5Stream)
            throws UnsupportedOperationException, IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        log.warn("Importing a Keras model from an InputStream pointing to contents of an HDF5 archive currently not supported.");
        throw new UnsupportedOperationException("Importing a Keras model from an InputStream currently not supported "
            + "because it is not possible to load an HDF5 file from a memory buffer using the HDF5 C++ API. "
            + "See: http://stackoverflow.com/questions/18449972/how-can-i-open-hdf5-file-from-memory-buffer-using-hdf5-c-api");

        /* One very hacky workaround would be to write the InputStream out to
         * a temporary file and then use the "from filename" constructor to
         * import from that file, as follows:
         *
         * File tempFile = File.createTempFile("temporary_model_archive",".h5");
         * tempFile.deleteOnExit();
         * tempFile.canWrite();
         * FileOutputStream tempOutputStream = new FileOutputStream(tempFile);
         * IOUtils.copy(modelHdf5Stream, tempOutputStream);
         * tempOutputStream.close();
         * String tempFilename = tempFile.getAbsolutePath();
         * super(tempFilename);
         */
    }

    /**
     * Constructor from HDF5 model archive.
     *
     * @param modelHdf5Filename     path to HDF5 archive storing Keras model
     * @throws IOException
     */
    public KerasModelImport(String modelHdf5Filename)
            throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        /* Open HDF5 archive model file. */
        hdf5.H5File file = new hdf5.H5File(modelHdf5Filename, H5F_ACC_RDONLY);

        /* Read model and training configurations from top-level attributes. */
        this.modelJson = readJsonStringFromHdf5Attribute(file, "model_config");
        this.modelClassName = getModelClassName(this.modelJson);
        this.trainingJson = readJsonStringFromHdf5Attribute(file, "training_config");

        /* Read weights from "/weights" group. */
        this.weights = readWeightsFromHdf5(file, "/model_weights");
        file.close();
    }

    /**
     * Constructor that takes filenames for JSON model configuration and for
     * HDF5 weights archive.
     *
     * @param modelJsonFilename       path to JSON file storing Keras Sequential model configuration
     * @param weightsHdf5Filename     path to HDF5 archive storing Keras model weights
     * @throws IOException
     */
    public KerasModelImport(String modelJsonFilename, String weightsHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        /* Read model configuration from JSON file. */
        this.modelJson = new String(Files.readAllBytes(Paths.get(modelJsonFilename)));
        this.modelClassName = getModelClassName(this.modelJson);

        /* Open HDF5 archive weights file. */
        hdf5.H5File file = new hdf5.H5File(weightsHdf5Filename, H5F_ACC_RDONLY);

        /* Read weights from root ("/") group. */
        this.weights = readWeightsFromHdf5(file, "/");
        file.close();
    }

    /**
     * Get model configuration JSON.
     *
     * @return
     */
    public String getModelJson() {
        return modelJson;
    }

    /**
     * Get training configuration JSON.
     *
     * @return
     */
    public String getTrainingJson() {
        return trainingJson;
    }

    /**
     * Get model class name (Model, Sequential, etc.).
     *
     * @return
     */
    public String getModelClassName() {
        return modelClassName;
    }

    /**
     * Get model weights stored as map from layer to parameter to INDArray.
     *
     * @return
     */
    public Map<String, Map<String, INDArray>> getWeights() {
        return weights;
    }

    /**
     * Read Keras model weights from specified HDF5 file and Group into a map
     * from layer to parameter to weights (INDArray).
     *
     * @param file                   open HDF5 archive file
     * @param weightsGroupName       name of root HDF5 Group storing all Keras weights for single model
     * @return              nested Map from layer names to parameter names to INDArrays
     */
    private static Map<String,Map<String,INDArray>> readWeightsFromHdf5(hdf5.H5File file, String weightsGroupName)
            throws UnsupportedKerasConfigurationException {
        hdf5.Group weightsGroup = file.asCommonFG().openGroup(weightsGroupName);

        Map<String,Map<String,INDArray>> weightsMap = new HashMap<String,Map<String,INDArray>>();

        List<hdf5.Group> groups = new ArrayList<hdf5.Group>();
        groups.add(weightsGroup);
        while (!groups.isEmpty()) {
            hdf5.Group g = groups.remove(0);
            for (int i = 0; i < g.asCommonFG().getNumObjs(); i++) {
                BytePointer objPtr = g.asCommonFG().getObjnameByIdx(i);
                String objName = objPtr.getString();
                int objType = g.asCommonFG().childObjType(objPtr);
                switch (objType) {
                    case H5O_TYPE_DATASET:
                        /* Keras parameter names are typically formatted as [layer name]_[layer no]_[parameter].
                         * For example, the weight matrix in the first Dense layer will be named "dense_1_W."
                         */
                        String[] tokens = objName.split("_");
                        String layerName = StringUtils.join(Arrays.copyOfRange(tokens, 0, 2), "_");
                        String paramName = StringUtils.join(Arrays.copyOfRange(tokens, 2, tokens.length), "_");
                        /* TensorFlow backend often appends ":" followed by one
                         * or more digits to parameter names, but this is not
                         * reflected in the model config. We must strip it off.
                         */
                        Pattern p = Pattern.compile(":\\d+$");
                        Matcher m = p.matcher(paramName);

                        hdf5.DataSet d = g.asCommonFG().openDataSet(objPtr);
                        hdf5.DataSpace space = d.getSpace();
                        int nbDims = (int)space.getSimpleExtentNdims();
                        long[] dims = new long[nbDims];
                        space.getSimpleExtentDims(dims);
                        float[] weightBuffer = null;
                        FloatPointer fp = null;
                        int j = 0;
                        INDArray weights = null;
                        if (m.find())
                            paramName = m.replaceFirst("");
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
                            case 1: /* Bias */
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
                                throw new UnsupportedKerasConfigurationException("Cannot import weights with rank " + nbDims);

                        }
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
     * Read contents of top-level string attribute from HDF5 File archive.
     *
     * @param file          HDF5 File
     * @param attribute     name of attribute
     * @return              contents of attribute as String
     */
    private static String readJsonStringFromHdf5Attribute(hdf5.H5File file, String attribute) throws InvalidKerasConfigurationException {
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
                throw new InvalidKerasConfigurationException("Could not read abnormally long Keras config. Please file an issue!");
            }
        }
        return jsonString;
    }

    /**
     * Convenience function for parsing JSON strings.
     *
     * @param modelJson    string containing valid JSON
     * @return             nested Map with arbitrary depth
     * @throws IOException
     */
    private static String getModelClassName(String modelJson) throws IOException, InvalidKerasConfigurationException {
        ObjectMapper mapper = new ObjectMapper();
        TypeReference<HashMap<String,Object>> typeRef = new TypeReference<HashMap<String,Object>>() {};
        Map<String,Object> modelConfig = mapper.readValue(modelJson, typeRef);
        if (!modelConfig.containsKey(MODEL_FIELD_CLASS_NAME))
            throw new InvalidKerasConfigurationException("Unable to determine Keras model class name.");
        return (String)modelConfig.get(MODEL_FIELD_CLASS_NAME);
    }
}
