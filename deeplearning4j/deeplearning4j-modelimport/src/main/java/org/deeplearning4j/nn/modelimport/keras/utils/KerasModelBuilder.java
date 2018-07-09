package org.deeplearning4j.nn.modelimport.keras.utils;

import lombok.Data;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.modelimport.keras.Hdf5Archive;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.modelimport.keras.config.KerasModelConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

@Data
public class KerasModelBuilder implements Cloneable, Closeable {
    protected String modelJson = null;
    protected String modelYaml = null;
    protected String trainingJson = null;
    protected String trainingYaml = null;
    protected Hdf5Archive weightsArchive = null;
    protected String weightsRoot = null;
    protected Hdf5Archive trainingArchive = null;
    protected boolean enforceTrainingConfig = false;
    protected KerasModelConfiguration config;
    protected int[] inputShape = null;
    protected KerasLayer.DimOrder dimOrder = null;


    /**
     * KerasModelBuilder constructed from a model configuration.
     *
     * @param config KerasModelConfiguration
     */
    public KerasModelBuilder(KerasModelConfiguration config) {
        this.config = config;
    }

    /**
     * Set model architecture from model JSON string.
     *
     * @param modelJson model as JSON string.
     * @return Model Builder
     */
    public KerasModelBuilder modelJson(String modelJson) {
        this.modelJson = modelJson;
        return this;
    }

    /**
     * Set model architecture from model YAML string.
     *
     * @param modelYaml model as YAML string.
     * @return Model Builder
     */
    public KerasModelBuilder modelYaml(String modelYaml) {
        this.modelYaml = modelYaml;
        return this;
    }

    /**
     * Set model architecture from file name pointing to model JSON string.
     *
     * @param modelJsonFilename Name of file containing model JSON string
     * @return Model Builder
     * @throws IOException I/O Exception
     */
    public KerasModelBuilder modelJsonFilename(String modelJsonFilename) throws IOException {
        checkForExistence(modelJsonFilename);
        this.modelJson = new String(Files.readAllBytes(Paths.get(modelJsonFilename)));
        return this;
    }

    /**
     * Set model architecture from file name pointing to model YAML string.
     *
     * @param modelYamlFilename Name of file containing model YAML string
     * @return Model Builder
     * @throws IOException I/O Exception
     */
    public KerasModelBuilder modelYamlFilename(String modelYamlFilename) throws IOException {
        checkForExistence(modelYamlFilename);
        this.modelJson = new String(Files.readAllBytes(Paths.get(modelYamlFilename)));
        return this;
    }

    /**
     * Set model architecture from input stream of model JSON.
     *
     * @param modelJsonInputStream Input stream of model JSON
     * @return Model builder
     * @throws IOException I/O exception
     */
    public KerasModelBuilder modelJsonInputStream(InputStream modelJsonInputStream) throws IOException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        IOUtils.copy(modelJsonInputStream, byteArrayOutputStream);
        this.modelJson = new String(byteArrayOutputStream.toByteArray());
        return this;
    }

    /**
     * Set model architecture from input stream of model YAML.
     *
     * @param modelYamlInputStream Input stream of model YAML
     * @return Model builder
     * @throws IOException I/O exception
     */
    public KerasModelBuilder modelYamlInputStream(InputStream modelYamlInputStream) throws IOException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        IOUtils.copy(modelYamlInputStream, byteArrayOutputStream);
        this.modelJson = new String(byteArrayOutputStream.toByteArray());
        return this;
    }

    /**
     * Provide input shape for Keras models that have been compiled without one. DL4J
     * needs this shape information on import to infer shapes of later layers and do
     * shape validation.
     *
     * @param inputShape Input shape as int array
     * @return Model Builder
     */
    public KerasModelBuilder inputShape(int[] inputShape) {
        this.inputShape = inputShape;
        return this;
    }

    /**
     * Provide training configuration as JSON string
     *
     * @param trainingJson Training JSON string
     * @return Model builder
     */
    public KerasModelBuilder trainingJson(String trainingJson) {
        this.trainingJson = trainingJson;
        return this;
    }

    /**
     * Provide training configuration as YAML string
     *
     * @param trainingYaml Training YAML string
     * @return Model builder
     */
    public KerasModelBuilder trainingYaml(String trainingYaml) {
        this.trainingYaml = trainingYaml;
        return this;
    }

    /**
     * Manually set dim order for Keras model, i.e. either TENSORFLOW (channels last)
     * or THEANO (channels first).
     *
     * Dim ordering will be automatically inferred from your model file, so don't
     * tamper with this option unless you're sure what you're doing. Explicitly
     * setting dim ordering can be useful for very old Keras models (before version 1.2),
     * for which inference can be difficult.
     *
     * @param dimOrder Ordering of dimensions (channels first vs. last)
     * @return Model builder
     */
    public KerasModelBuilder dimOrder(KerasLayer.DimOrder dimOrder){
        this.dimOrder = dimOrder;
        return this;
    }

    /**
     * Provide training configuration as file input stream from JSON
     *
     * @param trainingJsonInputStream Input stream of training JSON string
     * @return Model builder
     */
    public KerasModelBuilder trainingJsonInputStream(InputStream trainingJsonInputStream) throws IOException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        IOUtils.copy(trainingJsonInputStream, byteArrayOutputStream);
        this.trainingJson = new String(byteArrayOutputStream.toByteArray());
        return this;
    }

    /**
     * Provide training configuration as file input stream from YAML
     *
     * @param trainingYamlInputStream Input stream of training YAML string
     * @return Model builder
     */
    public KerasModelBuilder trainingYamlInputStream(InputStream trainingYamlInputStream) throws IOException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        IOUtils.copy(trainingYamlInputStream, byteArrayOutputStream);
        this.trainingYaml = new String(byteArrayOutputStream.toByteArray());
        return this;
    }

    /**
     * Set full model HDF5 (architecture, weights and training configuration) by providing the HDF5 filename.
     *
     * @param modelHdf5Filename File name of HDF5 file containing full model
     * @return Model builder
     * @throws UnsupportedKerasConfigurationException Unsupported configuration
     * @throws InvalidKerasConfigurationException Invalid configuration
     * @throws IOException I/O exception
     */
    public KerasModelBuilder modelHdf5Filename(String modelHdf5Filename)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException, IOException {
        checkForExistence(modelHdf5Filename);
        try {
            this.weightsArchive = this.trainingArchive = new Hdf5Archive(modelHdf5Filename);
            this.weightsRoot = config.getTrainingWeightsRoot();
            if (!this.weightsArchive.hasAttribute(config.getTrainingModelConfigAttribute()))
                throw new InvalidKerasConfigurationException(
                        "Model configuration attribute missing from " + modelHdf5Filename + " archive.");
            String initialModelJson = this.weightsArchive.readAttributeAsJson(
                    config.getTrainingModelConfigAttribute());

            String kerasVersion = this.weightsArchive.readAttributeAsFixedLengthString(
                    config.getFieldKerasVersion(), 5);
            Map<String, Object> modelMapper = KerasModelUtils.parseJsonString(initialModelJson);
            modelMapper.put(config.getFieldKerasVersion(), kerasVersion);

            int majorKerasVersion = Character.getNumericValue(kerasVersion.charAt(0));
            if (majorKerasVersion == 2) {
                String backend = this.weightsArchive.readAttributeAsString(config.getFieldBackend());
                modelMapper.put(config.getFieldBackend(), backend);
            }

            this.modelJson = new ObjectMapper().writeValueAsString(modelMapper);
            if (this.trainingArchive.hasAttribute(config.getTrainingTrainingConfigAttribute()))
                this.trainingJson = this.trainingArchive
                        .readAttributeAsJson(config.getTrainingTrainingConfigAttribute());
        } catch (Throwable t) {
            close();
            throw t;
        }
        return this;
    }

    /**
     * Set weights of the model by providing the file name of the corresponding weights HDF5 file.
     * The root of the HDF5 group containing weights won't be set by this method.
     *
     * @param weightsHdf5Filename File name of weights HDF5
     * @return Model builder
     */
    public KerasModelBuilder weightsHdf5FilenameNoRoot(String weightsHdf5Filename) throws IOException {
        checkForExistence(weightsHdf5Filename);
        this.weightsArchive = new Hdf5Archive(weightsHdf5Filename);
        return this;
    }

    /**
     * Set weights of the model by providing the file name of the corresponding weights HDF5 file.
     * The root of the HDF5 group containing weights will be read and set from the configuration of this
     * model builder instance.
     *
     * @param weightsHdf5Filename File name of weights HDF5
     * @return Model builder
     */
    public KerasModelBuilder weightsHdf5Filename(String weightsHdf5Filename) throws IOException {
        checkForExistence(weightsHdf5Filename);
        this.weightsArchive = new Hdf5Archive(weightsHdf5Filename);
        this.weightsRoot = config.getTrainingWeightsRoot();
        return this;
    }

    /**
     * Determine whether to enforce loading a training configuration or not.
     *
     * @param enforceTrainingConfig boolean, read training config or not
     * @return Model builder
     */
    public KerasModelBuilder enforceTrainingConfig(boolean enforceTrainingConfig) {
        this.enforceTrainingConfig = enforceTrainingConfig;
        return this;
    }

    /**
     * Build a KerasModel (corresponding to ComputationGraph) from this model builder.
     *
     * @return KerasModel
     * @throws IOException I/O exception
     * @throws InvalidKerasConfigurationException Invalid configuration
     * @throws UnsupportedKerasConfigurationException Unsupported configuration
     */
    public KerasModel buildModel()
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasModel model = new KerasModel(this);
        close();
        return model;
    }

    /**
     * Build a KerasSequentialModel (corresponding to MultiLayerNetwork) from this model builder.
     *
     * @return KerasSequentialModel
     * @throws IOException I/O exception
     * @throws InvalidKerasConfigurationException Invalid configuration
     * @throws UnsupportedKerasConfigurationException Unsupported configuration
     */
    public KerasSequentialModel buildSequential()
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasSequentialModel sequentialModel = new KerasSequentialModel(this);
        close();
        return sequentialModel;
    }

    /**
     * Close all HDF5 archives for this model builder.
     */
    @Override
    public void close() {
        if (trainingArchive != null && trainingArchive != weightsArchive) {
            trainingArchive.close();
            trainingArchive = null;
        }
        if (weightsArchive != null) {
            weightsArchive.close();
            weightsArchive = null;
        }
    }

    /**
     * Check if the file corresponding to model JSON/YAML or HDF5 files actually exists
     * and throw an explicit exception.
     *
     * @param fileName File name to check for existence
     * @throws FileNotFoundException File not found
     */
    private void checkForExistence(String fileName) throws IOException {
        File file = new File(fileName);
        if (!file.exists()) {
            throw new FileNotFoundException("File with name " + fileName + " does not exist.");
        }
        if (!file.isFile()) {
            throw new IOException("Provided string does not correspond to an actual file.");
        }

    }
}