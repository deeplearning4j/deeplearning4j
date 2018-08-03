/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.util;

import com.google.common.io.Files;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.output.CloseShieldOutputStream;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.reports.Task;
import org.nd4j.linalg.primitives.Pair;

import java.io.*;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * Utility class suited to save/restore neural net models
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class ModelSerializer {

    public static final String UPDATER_BIN = "updaterState.bin";
    public static final String NORMALIZER_BIN = "normalizer.bin";
    public static final String CONFIGURATION_JSON = "configuration.json";
    public static final String COEFFICIENTS_BIN = "coefficients.bin";
    public static final String NO_PARAMS_MARKER = "noParams.marker";
    public static final String PREPROCESSOR_BIN = "preprocessor.bin";

    private ModelSerializer() {}

    /**
     * Write a model to a file
     * @param model the model to write
     * @param file the file to write to
     * @param saveUpdater whether to save the updater or not
     * @throws IOException
     */
    public static void writeModel(@NonNull Model model, @NonNull File file, boolean saveUpdater) throws IOException {
        writeModel(model,file,saveUpdater,null);
    }



    /**
     * Write a model to a file
     * @param model the model to write
     * @param file the file to write to
     * @param saveUpdater whether to save the updater or not
     * @param dataNormalization the normalizer to save (optional)
     * @throws IOException
     */
    public static void writeModel(@NonNull Model model, @NonNull File file, boolean saveUpdater,DataNormalization dataNormalization) throws IOException {
        try (BufferedOutputStream stream = new BufferedOutputStream(new FileOutputStream(file))) {
            writeModel(model, stream, saveUpdater,dataNormalization);
        }
    }


    /**
     * Write a model to a file path
     * @param model the model to write
     * @param path the path to write to
     * @param saveUpdater whether to save the updater
     *                    or not
     * @throws IOException
     */
    public static void writeModel(@NonNull Model model, @NonNull String path, boolean saveUpdater) throws IOException {
        try (BufferedOutputStream stream = new BufferedOutputStream(new FileOutputStream(path))) {
            writeModel(model, stream, saveUpdater);
        }
    }

    /**
     * Write a model to an output stream
     * @param model the model to save
     * @param stream the output stream to write to
     * @param saveUpdater whether to save the updater for the model or not
     * @throws IOException
     */
    public static void writeModel(@NonNull Model model, @NonNull OutputStream stream, boolean saveUpdater)
            throws IOException {
        writeModel(model,stream,saveUpdater,null);
    }




    /**
     * Write a model to an output stream
     * @param model the model to save
     * @param stream the output stream to write to
     * @param saveUpdater whether to save the updater for the model or not
     * @param dataNormalization the normalizer ot save (may be null)
     * @throws IOException
     */
    public static void writeModel(@NonNull Model model, @NonNull OutputStream stream, boolean saveUpdater,DataNormalization dataNormalization)
            throws IOException {
        ZipOutputStream zipfile = new ZipOutputStream(new CloseShieldOutputStream(stream));

        // Save configuration as JSON
        String json = "";
        if (model instanceof MultiLayerNetwork) {
            json = ((MultiLayerNetwork) model).getLayerWiseConfigurations().toJson();
        } else if (model instanceof ComputationGraph) {
            json = ((ComputationGraph) model).getConfiguration().toJson();
        }
        ZipEntry config = new ZipEntry(CONFIGURATION_JSON);
        zipfile.putNextEntry(config);
        zipfile.write(json.getBytes());

        // Save parameters as binary
        ZipEntry coefficients = new ZipEntry(COEFFICIENTS_BIN);
        zipfile.putNextEntry(coefficients);
        DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(zipfile));
        INDArray params = model.params();
        if(params != null) {
            try {
                Nd4j.write(model.params(), dos);
            } finally {
                dos.flush();
            }
        } else {
            ZipEntry noParamsMarker = new ZipEntry(NO_PARAMS_MARKER);
            zipfile.putNextEntry(noParamsMarker);
        }

        if (saveUpdater) {
            INDArray updaterState = null;
            if (model instanceof MultiLayerNetwork) {
                updaterState = ((MultiLayerNetwork) model).getUpdater().getStateViewArray();
            } else if (model instanceof ComputationGraph) {
                updaterState = ((ComputationGraph) model).getUpdater().getStateViewArray();
            }

            if (updaterState != null && updaterState.length() > 0) {
                ZipEntry updater = new ZipEntry(UPDATER_BIN);
                zipfile.putNextEntry(updater);

                try {
                    Nd4j.write(updaterState, dos);
                } finally {
                    dos.flush();
                }
            }
        }


        if(dataNormalization != null) {
            // now, add our normalizer as additional entry
            ZipEntry nEntry = new ZipEntry(NORMALIZER_BIN);
            zipfile.putNextEntry(nEntry);
            NormalizerSerializer.getDefault().write(dataNormalization, zipfile);
        }

        dos.close();
        zipfile.close();
    }

    /**
     * Load a multi layer network from a file
     *
     * @param file the file to load from
     * @return the loaded multi layer network
     * @throws IOException
     */
    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull File file) throws IOException {
        return restoreMultiLayerNetwork(file, true);
    }


    /**
     * Load a multi layer network from a file
     *
     * @param file the file to load from
     * @return the loaded multi layer network
     * @throws IOException
     */
    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull File file, boolean loadUpdater)
            throws IOException {
        ZipFile zipFile = new ZipFile(file);

        boolean gotConfig = false;
        boolean gotCoefficients = false;
        boolean gotUpdaterState = false;
        boolean gotPreProcessor = false;

        String json = "";
        INDArray params = null;
        Updater updater = null;
        INDArray updaterState = null;
        DataSetPreProcessor preProcessor = null;


        ZipEntry config = zipFile.getEntry(CONFIGURATION_JSON);
        if (config != null) {
            //restoring configuration

            InputStream stream = zipFile.getInputStream(config);
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
            String line = "";
            StringBuilder js = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                js.append(line).append("\n");
            }
            json = js.toString();

            reader.close();
            stream.close();
            gotConfig = true;
        }


        ZipEntry coefficients = zipFile.getEntry(COEFFICIENTS_BIN);
        if (coefficients != null ) {
            if(coefficients.getSize() > 0) {
                InputStream stream = zipFile.getInputStream(coefficients);
                DataInputStream dis = new DataInputStream(new BufferedInputStream(stream));
                params = Nd4j.read(dis);

                dis.close();
                gotCoefficients = true;
            } else {
                ZipEntry noParamsMarker = zipFile.getEntry(NO_PARAMS_MARKER);
                gotCoefficients = (noParamsMarker != null);
            }
        }

        if (loadUpdater) {
            ZipEntry updaterStateEntry = zipFile.getEntry(UPDATER_BIN);
            if (updaterStateEntry != null) {
                InputStream stream = zipFile.getInputStream(updaterStateEntry);
                DataInputStream dis = new DataInputStream(new BufferedInputStream(stream));
                updaterState = Nd4j.read(dis);

                dis.close();
                gotUpdaterState = true;
            }
        }

        ZipEntry prep = zipFile.getEntry(PREPROCESSOR_BIN);
        if (prep != null) {
            InputStream stream = zipFile.getInputStream(prep);
            ObjectInputStream ois = new ObjectInputStream(stream);

            try {
                preProcessor = (DataSetPreProcessor) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e);
            }

            gotPreProcessor = true;
        }


        zipFile.close();

        if (gotConfig && gotCoefficients) {
            MultiLayerConfiguration confFromJson;
            try{
               confFromJson = MultiLayerConfiguration.fromJson(json);
            } catch (Exception e){
                ComputationGraphConfiguration cg;
                try{
                    cg = ComputationGraphConfiguration.fromJson(json);
                } catch (Exception e2){
                    //Invalid, and not a compgraph
                    throw new RuntimeException("Error deserializing JSON MultiLayerConfiguration. Saved model JSON is" +
                            " not a valid MultiLayerConfiguration", e);
                }
                if(cg.getNetworkInputs() != null && cg.getVertices() != null) {
                    throw new RuntimeException("Error deserializing JSON MultiLayerConfiguration. Saved model appears to be " +
                            "a ComputationGraph - use ModelSerializer.restoreComputationGraph instead");
                } else {
                    throw e;
                }
            }
            MultiLayerNetwork network = new MultiLayerNetwork(confFromJson);
            network.init(params, false);

            if (gotUpdaterState && updaterState != null) {
                network.getUpdater().setStateViewArray(network, updaterState, false);
            }
            return network;
        } else
            throw new IllegalStateException("Model wasnt found within file: gotConfig: [" + gotConfig
                    + "], gotCoefficients: [" + gotCoefficients + "], gotUpdater: [" + gotUpdaterState + "]");
    }


    /**
     * Load a MultiLayerNetwork from InputStream from an input stream<br>
     * Note: the input stream is read fully and closed by this method. Consequently, the input stream cannot be re-used.
     *
     * @param is the inputstream to load from
     * @return the loaded multi layer network
     * @throws IOException
     * @see #restoreMultiLayerNetworkAndNormalizer(InputStream, boolean)
     */
    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull InputStream is, boolean loadUpdater)
            throws IOException {
        checkInputStream(is);

        File tmpFile = null;
        try{
            tmpFile = tempFileFromStream(is);
            return restoreMultiLayerNetwork(tmpFile, loadUpdater);
        } finally {
            if(tmpFile != null){
                tmpFile.delete();
            }
        }
    }

    /**
     * Restore a multi layer network from an input stream<br>
     * * Note: the input stream is read fully and closed by this method. Consequently, the input stream cannot be re-used.
     *
     *
     * @param is the input stream to restore from
     * @return the loaded multi layer network
     * @throws IOException
     * @see #restoreMultiLayerNetworkAndNormalizer(InputStream, boolean)
     */
    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull InputStream is) throws IOException {
        return restoreMultiLayerNetwork(is, true);
    }

    /**
     * Load a MultilayerNetwork model from a file
     *
     * @param path path to the model file, to get the computation graph from
     * @return the loaded computation graph
     *
     * @throws IOException
     */
    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull String path) throws IOException {
        return restoreMultiLayerNetwork(new File(path), true);
    }

    /**
     * Load a MultilayerNetwork model from a file
     * @param path path to the model file, to get the computation graph from
     * @return the loaded computation graph
     *
     * @throws IOException
     */
    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull String path, boolean loadUpdater)
            throws IOException {
        return restoreMultiLayerNetwork(new File(path), loadUpdater);
    }

    /**
     * Restore a MultiLayerNetwork and Normalizer (if present - null if not) from the InputStream.
     * Note: the input stream is read fully and closed by this method. Consequently, the input stream cannot be re-used.
     *
     * @param is          Input stream to read from
     * @param loadUpdater Whether to load the updater from the model or not
     * @return Model and normalizer, if present
     * @throws IOException If an error occurs when reading from the stream
     */
    public static Pair<MultiLayerNetwork, Normalizer> restoreMultiLayerNetworkAndNormalizer(
            @NonNull InputStream is, boolean loadUpdater) throws IOException {
        checkInputStream(is);

        File tmpFile = null;
        try {
            tmpFile = tempFileFromStream(is);
            return restoreMultiLayerNetworkAndNormalizer(tmpFile, loadUpdater);
        } finally {
            if (tmpFile != null) {
                tmpFile.delete();
            }
        }
    }

    /**
     * Restore a MultiLayerNetwork and Normalizer (if present - null if not) from a File
     *
     * @param file        File to read the model and normalizer from
     * @param loadUpdater Whether to load the updater from the model or not
     * @return Model and normalizer, if present
     * @throws IOException If an error occurs when reading from the File
     */
    public static Pair<MultiLayerNetwork, Normalizer> restoreMultiLayerNetworkAndNormalizer(@NonNull File file, boolean loadUpdater)
            throws IOException {
        MultiLayerNetwork net = restoreMultiLayerNetwork(file, loadUpdater);
        Normalizer norm = restoreNormalizerFromFile(file);
        return new Pair<>(net, norm);
    }

    /**
     * Load a computation graph from a file
     * @param path path to the model file, to get the computation graph from
     * @return the loaded computation graph
     *
     * @throws IOException
     */
    public static ComputationGraph restoreComputationGraph(@NonNull String path) throws IOException {
        return restoreComputationGraph(new File(path), true);
    }

    /**
     * Load a computation graph from a file
     * @param path path to the model file, to get the computation graph from
     * @return the loaded computation graph
     *
     * @throws IOException
     */
    public static ComputationGraph restoreComputationGraph(@NonNull String path, boolean loadUpdater)
            throws IOException {
        return restoreComputationGraph(new File(path), loadUpdater);
    }


    /**
     * Load a computation graph from a InputStream
     * @param is the inputstream to get the computation graph from
     * @return the loaded computation graph
     *
     * @throws IOException
     */
    public static ComputationGraph restoreComputationGraph(@NonNull InputStream is, boolean loadUpdater)
            throws IOException {
        checkInputStream(is);

        File tmpFile = null;
        try{
            tmpFile = tempFileFromStream(is);
            return restoreComputationGraph(tmpFile, loadUpdater);
        } finally {
            if(tmpFile != null){
                tmpFile.delete();
            }
        }
    }

    /**
     * Load a computation graph from a InputStream
     * @param is the inputstream to get the computation graph from
     * @return the loaded computation graph
     *
     * @throws IOException
     */
    public static ComputationGraph restoreComputationGraph(@NonNull InputStream is) throws IOException {
        return restoreComputationGraph(is, true);
    }

    /**
     * Load a computation graph from a file
     * @param file the file to get the computation graph from
     * @return the loaded computation graph
     *
     * @throws IOException
     */
    public static ComputationGraph restoreComputationGraph(@NonNull File file) throws IOException {
        return restoreComputationGraph(file, true);
    }

    /**
     * Restore a ComputationGraph and Normalizer (if present - null if not) from the InputStream.
     * Note: the input stream is read fully and closed by this method. Consequently, the input stream cannot be re-used.
     *
     * @param is          Input stream to read from
     * @param loadUpdater Whether to load the updater from the model or not
     * @return Model and normalizer, if present
     * @throws IOException If an error occurs when reading from the stream
     */
    public static Pair<ComputationGraph, Normalizer> restoreComputationGraphAndNormalizer(
            @NonNull InputStream is, boolean loadUpdater) throws IOException {
        checkInputStream(is);

        File tmpFile = null;
        try {
            tmpFile = tempFileFromStream(is);
            return restoreComputationGraphAndNormalizer(tmpFile, loadUpdater);
        } finally {
            if (tmpFile != null) {
                tmpFile.delete();
            }
        }
    }

    /**
     * Restore a ComputationGraph and Normalizer (if present - null if not) from a File
     *
     * @param file        File to read the model and normalizer from
     * @param loadUpdater Whether to load the updater from the model or not
     * @return Model and normalizer, if present
     * @throws IOException If an error occurs when reading from the File
     */
    public static Pair<ComputationGraph, Normalizer> restoreComputationGraphAndNormalizer(@NonNull File file, boolean loadUpdater)
            throws IOException {
        ComputationGraph net = restoreComputationGraph(file, loadUpdater);
        Normalizer norm = restoreNormalizerFromFile(file);
        return new Pair<>(net, norm);
    }

    /**
     * Load a computation graph from a file
     * @param file the file to get the computation graph from
     * @return the loaded computation graph
     *
     * @throws IOException
     */
    public static ComputationGraph restoreComputationGraph(@NonNull File file, boolean loadUpdater) throws IOException {
        ZipFile zipFile = new ZipFile(file);

        boolean gotConfig = false;
        boolean gotCoefficients = false;
        boolean gotUpdaterState = false;
        boolean gotPreProcessor = false;

        String json = "";
        INDArray params = null;
        INDArray updaterState = null;
        DataSetPreProcessor preProcessor = null;


        ZipEntry config = zipFile.getEntry(CONFIGURATION_JSON);
        if (config != null) {
            //restoring configuration

            InputStream stream = zipFile.getInputStream(config);
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
            String line = "";
            StringBuilder js = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                js.append(line).append("\n");
            }
            json = js.toString();

            reader.close();
            stream.close();
            gotConfig = true;
        }


        ZipEntry coefficients = zipFile.getEntry(COEFFICIENTS_BIN);
        if (coefficients != null) {
            if(coefficients.getSize() > 0) {
                InputStream stream = zipFile.getInputStream(coefficients);
                DataInputStream dis = new DataInputStream(new BufferedInputStream(stream));
                params = Nd4j.read(dis);

                dis.close();
                gotCoefficients = true;
            } else {
                ZipEntry noParamsMarker = zipFile.getEntry(NO_PARAMS_MARKER);
                gotCoefficients = (noParamsMarker != null);
            }
        }


        if (loadUpdater) {
            ZipEntry updaterStateEntry = zipFile.getEntry(UPDATER_BIN);
            if (updaterStateEntry != null) {
                InputStream stream = zipFile.getInputStream(updaterStateEntry);
                DataInputStream dis = new DataInputStream(new BufferedInputStream(stream));
                updaterState = Nd4j.read(dis);

                dis.close();
                gotUpdaterState = true;
            }
        }

        ZipEntry prep = zipFile.getEntry(PREPROCESSOR_BIN);
        if (prep != null) {
            InputStream stream = zipFile.getInputStream(prep);
            ObjectInputStream ois = new ObjectInputStream(stream);

            try {
                preProcessor = (DataSetPreProcessor) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e);
            }

            gotPreProcessor = true;
        }


        zipFile.close();

        if (gotConfig && gotCoefficients) {
            ComputationGraphConfiguration confFromJson;
            try{
                confFromJson = ComputationGraphConfiguration.fromJson(json);
                if(confFromJson.getNetworkInputs() == null && (confFromJson.getVertices() == null || confFromJson.getVertices().size() == 0)){
                    //May be deserialized correctly, but mostly with null fields
                    throw new RuntimeException("Invalid JSON - not a ComputationGraphConfiguration");
                }
            } catch (Exception e){
                if(e.getMessage() != null && e.getMessage().contains("registerLegacyCustomClassesForJSON")){
                    throw e;
                }
                try{
                    MultiLayerConfiguration.fromJson(json);
                } catch (Exception e2){
                    //Invalid, and not a compgraph
                    throw new RuntimeException("Error deserializing JSON ComputationGraphConfiguration. Saved model JSON is" +
                            " not a valid ComputationGraphConfiguration", e);
                }
                throw new RuntimeException("Error deserializing JSON ComputationGraphConfiguration. Saved model appears to be " +
                        "a MultiLayerNetwork - use ModelSerializer.restoreMultiLayerNetwork instead");
            }
            ComputationGraph cg = new ComputationGraph(confFromJson);
            cg.init(params, false);


            if (gotUpdaterState && updaterState != null) {
                cg.getUpdater().setStateViewArray(updaterState);
            }
            return cg;
        } else
            throw new IllegalStateException("Model wasnt found within file: gotConfig: [" + gotConfig
                    + "], gotCoefficients: [" + gotCoefficients + "], gotUpdater: [" + gotUpdaterState + "]");
    }

    /**
     *
     * @param model
     * @return
     */
    public static Task taskByModel(Model model) {
        Task task = new Task();
        try {
            task.setArchitectureType(Task.ArchitectureType.RECURRENT);
            if (model instanceof ComputationGraph) {
                task.setNetworkType(Task.NetworkType.ComputationalGraph);
                ComputationGraph network = (ComputationGraph) model;
                try {
                    if (network.getLayers() != null && network.getLayers().length > 0) {
                        for (Layer layer : network.getLayers()) {
                            if (layer.type().equals(Layer.Type.CONVOLUTIONAL)) {
                                task.setArchitectureType(Task.ArchitectureType.CONVOLUTION);
                                break;
                            } else if (layer.type().equals(Layer.Type.RECURRENT)
                                    || layer.type().equals(Layer.Type.RECURSIVE)) {
                                task.setArchitectureType(Task.ArchitectureType.RECURRENT);
                                break;
                            }
                        }
                    } else
                        task.setArchitectureType(Task.ArchitectureType.UNKNOWN);
                } catch (Exception e) {
                    // do nothing here
                }
            } else if (model instanceof MultiLayerNetwork) {
                task.setNetworkType(Task.NetworkType.MultilayerNetwork);
                MultiLayerNetwork network = (MultiLayerNetwork) model;
                try {
                    if (network.getLayers() != null && network.getLayers().length > 0) {
                        for (Layer layer : network.getLayers()) {
                            if (layer.type().equals(Layer.Type.CONVOLUTIONAL)) {
                                task.setArchitectureType(Task.ArchitectureType.CONVOLUTION);
                                break;
                            } else if (layer.type().equals(Layer.Type.RECURRENT)
                                    || layer.type().equals(Layer.Type.RECURSIVE)) {
                                task.setArchitectureType(Task.ArchitectureType.RECURRENT);
                                break;
                            }
                        }
                    } else
                        task.setArchitectureType(Task.ArchitectureType.UNKNOWN);
                } catch (Exception e) {
                    // do nothing here
                }
            }
            return task;
        } catch (Exception e) {
            task.setArchitectureType(Task.ArchitectureType.UNKNOWN);
            task.setNetworkType(Task.NetworkType.DenseNetwork);
            return task;
        }
    }

    /**
     * This method appends normalizer to a given persisted model.
     *
     * PLEASE NOTE: File should be model file saved earlier with ModelSerializer
     *
     * @param f
     * @param normalizer
     */
    public static void addNormalizerToModel(File f, Normalizer<?> normalizer) {
        File tempFile = null;
        try {
            // copy existing model to temporary file
            tempFile = File.createTempFile("tempcopy", "temp");
            tempFile.deleteOnExit();
            Files.copy(f, tempFile);
            try (ZipFile zipFile = new ZipFile(tempFile);
                 ZipOutputStream writeFile =
                         new ZipOutputStream(new BufferedOutputStream(new FileOutputStream(f)))) {
                // roll over existing files within model, and copy them one by one
                Enumeration<? extends ZipEntry> entries = zipFile.entries();
                while (entries.hasMoreElements()) {
                    ZipEntry entry = entries.nextElement();

                    // we're NOT copying existing normalizer, if any
                    if (entry.getName().equalsIgnoreCase(NORMALIZER_BIN))
                        continue;

                    log.debug("Copying: {}", entry.getName());

                    InputStream is = zipFile.getInputStream(entry);

                    ZipEntry wEntry = new ZipEntry(entry.getName());
                    writeFile.putNextEntry(wEntry);

                    IOUtils.copy(is, writeFile);
                }
                // now, add our normalizer as additional entry
                ZipEntry nEntry = new ZipEntry(NORMALIZER_BIN);
                writeFile.putNextEntry(nEntry);

                NormalizerSerializer.getDefault().write(normalizer, writeFile);
            }
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        } finally {
            if (tempFile != null) {
                tempFile.delete();
            }
        }
    }

    /**
     * Add an object to the (already existing) model file using Java Object Serialization. Objects can be restored
     * using {@link #getObjectFromFile(File, String)}
     * @param f   File to add the object to
     * @param key Key to store the object under
     * @param o   Object to store using Java object serialization
     */
    public static void addObjectToFile(@NonNull File f, @NonNull String key, @NonNull Object o){
        Preconditions.checkState(f.exists(), "File must exist: %s", f);
        Preconditions.checkArgument(!(UPDATER_BIN.equalsIgnoreCase(key) || NORMALIZER_BIN.equalsIgnoreCase(key)
                || CONFIGURATION_JSON.equalsIgnoreCase(key) || COEFFICIENTS_BIN.equalsIgnoreCase(key)
                || NO_PARAMS_MARKER.equalsIgnoreCase(key) || PREPROCESSOR_BIN.equalsIgnoreCase(key)),
                "Invalid key: Key is reserved for internal use: \"%s\"", key);
        File tempFile = null;
        try {
            // copy existing model to temporary file
            tempFile = File.createTempFile("tempcopy", "temp");
            tempFile.deleteOnExit();
            Files.copy(f, tempFile);
            f.delete();
            try (ZipFile zipFile = new ZipFile(tempFile);
                 ZipOutputStream writeFile =
                         new ZipOutputStream(new BufferedOutputStream(new FileOutputStream(f)))) {
                // roll over existing files within model, and copy them one by one
                Enumeration<? extends ZipEntry> entries = zipFile.entries();
                while (entries.hasMoreElements()) {
                    ZipEntry entry = entries.nextElement();

                    log.debug("Copying: {}", entry.getName());

                    InputStream is = zipFile.getInputStream(entry);

                    ZipEntry wEntry = new ZipEntry(entry.getName());
                    writeFile.putNextEntry(wEntry);

                    IOUtils.copy(is, writeFile);
                    writeFile.closeEntry();
                    is.close();
                }

                //Add new object:
                ZipEntry entry = new ZipEntry("objects/" + key);
                writeFile.putNextEntry(entry);

                try(ByteArrayOutputStream baos = new ByteArrayOutputStream(); ObjectOutputStream oos = new ObjectOutputStream(baos)){
                    oos.writeObject(o);
                    byte[] bytes = baos.toByteArray();
                    writeFile.write(bytes);
                }
                writeFile.closeEntry();

                writeFile.close();
                zipFile.close();

            }
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        } finally {
            if (tempFile != null) {
                tempFile.delete();
            }
        }
    }

    /**
     * Get an object with the specified key from the model file, that was previously added to the file using
     * {@link #addObjectToFile(File, String, Object)}
     *
     * @param f   model file to add the object to
     * @param key Key for the object
     * @param <T> Type of the object
     * @return The serialized object
     * @see #listObjectsInFile(File)
     */
    public static <T> T getObjectFromFile(@NonNull File f, @NonNull String key){
        Preconditions.checkState(f.exists(), "File must exist: %s", f);
        Preconditions.checkArgument(!(UPDATER_BIN.equalsIgnoreCase(key) || NORMALIZER_BIN.equalsIgnoreCase(key)
                        || CONFIGURATION_JSON.equalsIgnoreCase(key) || COEFFICIENTS_BIN.equalsIgnoreCase(key)
                        || NO_PARAMS_MARKER.equalsIgnoreCase(key) || PREPROCESSOR_BIN.equalsIgnoreCase(key)),
                "Invalid key: Key is reserved for internal use: \"%s\"", key);

        try (ZipFile zipFile = new ZipFile(f)) {
            ZipEntry entry = zipFile.getEntry("objects/" + key);
            if(entry == null){
                throw new IllegalStateException("No object with key \"" + key + "\" found");
            }

            Object o;
            try(ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(zipFile.getInputStream(entry)))){
                o = ois.readObject();
            }
            zipFile.close();
            return (T)o;
        } catch (IOException | ClassNotFoundException e){
            throw new RuntimeException("Error reading object (key = " + key + ") from file " + f, e);
        }
    }

    /**
     * List the keys of all objects added using the method {@link #addObjectToFile(File, String, Object)}
     * @param f File previously created with ModelSerializer
     * @return List of keys that can be used with {@link #getObjectFromFile(File, String)}
     */
    public static List<String> listObjectsInFile(@NonNull File f){
        Preconditions.checkState(f.exists(), "File must exist: %s", f);

        List<String> out = new ArrayList<>();
        try (ZipFile zipFile = new ZipFile(f)){

            Enumeration<? extends ZipEntry> entries = zipFile.entries();
            while(entries.hasMoreElements()){
                ZipEntry e = entries.nextElement();
                String name = e.getName();
                if(!e.isDirectory() && name.startsWith("objects/")){
                    String s = name.substring(8);
                    out.add(s);
                }
            }
            return out;
        } catch (IOException e){
            throw new RuntimeException(e);
        }
    }



    /**
     * This method restores normalizer from a given persisted model file
     *
     * PLEASE NOTE: File should be model file saved earlier with ModelSerializer with addNormalizerToModel being called
     *
     * @param file
     * @return
     */
    public static <T extends Normalizer> T restoreNormalizerFromFile(File file) {
        try (ZipFile zipFile = new ZipFile(file)) {
            ZipEntry norm = zipFile.getEntry(NORMALIZER_BIN);

            // checking for file existence
            if (norm == null)
                return null;

            return NormalizerSerializer.getDefault().restore(zipFile.getInputStream(norm));
        } catch (Exception e) {
            log.warn("Error while restoring normalizer, trying to restore assuming deprecated format...");
            DataNormalization restoredDeprecated = restoreNormalizerFromFileDeprecated(file);

            log.warn("Recovered using deprecated method. Will now re-save the normalizer to fix this issue.");
            addNormalizerToModel(file, restoredDeprecated);

            return (T) restoredDeprecated;
        }
    }


    /**
     * This method restores the normalizer form a persisted model file.
     *
     * @param is A stream to load data from.
     * @return the loaded normalizer
     */
    public static <T extends Normalizer> T restoreNormalizerFromInputStream(InputStream is) throws IOException {
        checkInputStream(is);

        File tmpFile = null;
        try {
            tmpFile = tempFileFromStream(is);
            return restoreNormalizerFromFile(tmpFile);
        } finally {
            if(tmpFile != null){
                tmpFile.delete();
            }
        }
    }

    /**
     * @deprecated
     *
     * This method restores normalizer from a given persisted model file serialized with Java object serialization
     *
     * @param file
     * @return
     */
    private static DataNormalization restoreNormalizerFromFileDeprecated(File file) {
        try (ZipFile zipFile = new ZipFile(file)) {
            ZipEntry norm = zipFile.getEntry(NORMALIZER_BIN);

            // checking for file existence
            if (norm == null)
                return null;

            InputStream stream = zipFile.getInputStream(norm);
            ObjectInputStream ois = new ObjectInputStream(stream);

            try {
                DataNormalization normalizer = (DataNormalization) ois.readObject();
                return normalizer;
            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    private static void checkInputStream(InputStream inputStream) throws IOException {

        /*
        //available method can return 0 in some cases: https://github.com/deeplearning4j/deeplearning4j/issues/4887
        int available;
        try{
            //InputStream.available(): A subclass' implementation of this method may choose to throw an IOException
            // if this input stream has been closed by invoking the close() method.
            available = inputStream.available();
        } catch (IOException e){
            throw new IOException("Cannot read from stream: stream may have been closed or is attempting to be read from" +
                    "multiple times?", e);
        }
        if(available <= 0){
            throw new IOException("Cannot read from stream: stream may have been closed or is attempting to be read from" +
                    "multiple times?");
        }
        */
    }

    private static void checkTempFileFromInputStream(File f) throws IOException {
        if (f.length() <= 0) {
            throw new IOException("Error reading from input stream: temporary file is empty after copying entire stream." +
                    " Stream may have been closed before reading, is attempting to be used multiple times, or does not" +
                    " point to a model file?");
        }
    }

    private static File tempFileFromStream(InputStream is) throws IOException{
        checkInputStream(is);
        File tmpFile = File.createTempFile("dl4jModelSerializer", "bin");
        try {
            tmpFile.deleteOnExit();
            BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(new FileOutputStream(tmpFile));
            IOUtils.copy(is, bufferedOutputStream);
            bufferedOutputStream.flush();
            IOUtils.closeQuietly(bufferedOutputStream);
            checkTempFileFromInputStream(tmpFile);
            return tmpFile;
        } catch (IOException e){
            if(tmpFile != null){
                tmpFile.delete();
            }
            throw e;
        }
    }
}
