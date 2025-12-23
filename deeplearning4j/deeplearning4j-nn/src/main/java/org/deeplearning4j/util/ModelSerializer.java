/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.util;

import org.apache.commons.io.input.CloseShieldInputStream;
import org.nd4j.common.util.ND4JFileUtils;
import org.nd4j.shade.guava.io.Files;
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
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

import java.io.*;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

@Slf4j
public class ModelSerializer {

    public static final String UPDATER_BIN = "updaterState.bin";
    public static final String NORMALIZER_BIN = "normalizer.bin";
    public static final String CONFIGURATION_JSON = "configuration.json";
    public static final String COEFFICIENTS_BIN = "coefficients.bin";
    public static final String NO_PARAMS_MARKER = "noParams.marker";
    public static final String PREPROCESSOR_BIN = "preprocessor.bin";

    /**
     * Maximum total decompressed size allowed when loading ZIP files (default: 1GB).
     * This limit protects against zip bomb attacks where small compressed files
     * expand to consume all available memory.
     * Can be overridden via system property "dl4j.model.maxZipSize" (in bytes).
     */
    public static final long DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE = 1024L * 1024L * 1024L; // 1GB

    /**
     * Maximum allowed compression ratio (uncompressed/compressed size).
     * Files with higher ratios are rejected as potential zip bombs.
     * Default is 100:1. Can be overridden via system property "dl4j.model.maxCompressionRatio".
     */
    public static final double DEFAULT_MAX_COMPRESSION_RATIO = 100.0;

    /**
     * Maximum number of entries allowed in a model ZIP file.
     * Protects against zip bombs with many small entries.
     * Can be overridden via system property "dl4j.model.maxZipEntries".
     */
    public static final int DEFAULT_MAX_ZIP_ENTRIES = 100;

    private static long maxTotalUncompressedSize = getConfiguredMaxSize();
    private static double maxCompressionRatio = getConfiguredMaxRatio();
    private static int maxZipEntries = getConfiguredMaxEntries();

    private static long getConfiguredMaxSize() {
        String prop = System.getProperty("dl4j.model.maxZipSize");
        if (prop != null) {
            try {
                return Long.parseLong(prop);
            } catch (NumberFormatException e) {
                log.warn("Invalid value for dl4j.model.maxZipSize: {}, using default", prop);
            }
        }
        return DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE;
    }

    private static double getConfiguredMaxRatio() {
        String prop = System.getProperty("dl4j.model.maxCompressionRatio");
        if (prop != null) {
            try {
                return Double.parseDouble(prop);
            } catch (NumberFormatException e) {
                log.warn("Invalid value for dl4j.model.maxCompressionRatio: {}, using default", prop);
            }
        }
        return DEFAULT_MAX_COMPRESSION_RATIO;
    }

    private static int getConfiguredMaxEntries() {
        String prop = System.getProperty("dl4j.model.maxZipEntries");
        if (prop != null) {
            try {
                return Integer.parseInt(prop);
            } catch (NumberFormatException e) {
                log.warn("Invalid value for dl4j.model.maxZipEntries: {}, using default", prop);
            }
        }
        return DEFAULT_MAX_ZIP_ENTRIES;
    }

    /**
     * Set the maximum total uncompressed size allowed when loading model ZIP files.
     * @param maxSize Maximum size in bytes (must be positive)
     */
    public static void setMaxTotalUncompressedSize(long maxSize) {
        Preconditions.checkArgument(maxSize > 0, "Max size must be positive, got %s", maxSize);
        maxTotalUncompressedSize = maxSize;
    }

    /**
     * Set the maximum compression ratio allowed for ZIP entries.
     * @param maxRatio Maximum ratio (must be >= 1.0)
     */
    public static void setMaxCompressionRatio(double maxRatio) {
        Preconditions.checkArgument(maxRatio >= 1.0, "Max ratio must be >= 1.0, got %s", maxRatio);
        maxCompressionRatio = maxRatio;
    }

    /**
     * Set the maximum number of entries allowed in model ZIP files.
     * @param maxEntries Maximum entries (must be positive)
     */
    public static void setMaxZipEntries(int maxEntries) {
        Preconditions.checkArgument(maxEntries > 0, "Max entries must be positive, got %s", maxEntries);
        maxZipEntries = maxEntries;
    }

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
        try (ZipOutputStream zipfile = new ZipOutputStream(new CloseShieldOutputStream(stream));
             DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(zipfile))) {

            // Save configuration as JSON
            String json = "";
            if (model instanceof MultiLayerNetwork) {
                json = ((MultiLayerNetwork) model).getLayerWiseConfigurations().toJson();
            } else if (model instanceof ComputationGraph) {
                json = ((ComputationGraph) model).getConfiguration().toJson();
            }

            ZipEntry config = new ZipEntry(CONFIGURATION_JSON);
            zipfile.putNextEntry(config);
            dos.write(json.getBytes());

            // Save parameters as binary
            ZipEntry coefficients = new ZipEntry(COEFFICIENTS_BIN);
            zipfile.putNextEntry(coefficients);
            INDArray params = model.params();
            if (params != null) {
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


            if (dataNormalization != null) {
                // now, add our normalizer as additional entry
                ZipEntry nEntry = new ZipEntry(NORMALIZER_BIN);
                zipfile.putNextEntry(nEntry);
                NormalizerSerializer.getDefault().write(dataNormalization, dos);
            }
        }
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
        try(InputStream is = new BufferedInputStream(new FileInputStream(file))){
            return restoreMultiLayerNetwork(is, loadUpdater);
        }
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
        return restoreMultiLayerNetworkHelper(is, loadUpdater).getFirst();
    }

    private static Pair<MultiLayerNetwork, Map<String,byte[]>> restoreMultiLayerNetworkHelper(@NonNull InputStream is, boolean loadUpdater)
            throws IOException {
        checkInputStream(is);

        Map<String, byte[]> zipFile = loadZipData(is);

        boolean gotConfig = false;
        boolean gotCoefficients = false;
        boolean gotUpdaterState = false;
        boolean gotPreProcessor = false;

        String json = "";
        INDArray params = null;
        Updater updater = null;
        INDArray updaterState = null;
        DataSetPreProcessor preProcessor = null;


        byte[] config = zipFile.get(CONFIGURATION_JSON);
        if (config != null) {
            //restoring configuration

            InputStream stream = new ByteArrayInputStream(config);
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


        byte[] coefficients = zipFile.get(COEFFICIENTS_BIN);
        if (coefficients != null ) {
            if(coefficients.length > 0) {
                InputStream stream = new ByteArrayInputStream(coefficients);
                DataInputStream dis = new DataInputStream(new BufferedInputStream(stream));
                params = Nd4j.read(dis);

                dis.close();
                gotCoefficients = true;
            } else {
                byte[] noParamsMarker = zipFile.get(NO_PARAMS_MARKER);
                gotCoefficients = (noParamsMarker != null);
            }
        }

        if (loadUpdater) {
        	byte[] updaterStateEntry = zipFile.get(UPDATER_BIN);
            if (updaterStateEntry != null) {
                InputStream stream = new ByteArrayInputStream(updaterStateEntry);
                DataInputStream dis = new DataInputStream(new BufferedInputStream(stream));
                updaterState = Nd4j.read(dis);

                dis.close();
                gotUpdaterState = true;
            }
        }

        byte[] prep = zipFile.get(PREPROCESSOR_BIN);
        if (prep != null) {
            InputStream stream = new ByteArrayInputStream(prep);
            ObjectInputStream ois = new ObjectInputStream(stream);

            try {
                preProcessor = (DataSetPreProcessor) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e);
            }

            gotPreProcessor = true;
        }



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

            //Handle legacy config - no network DataType in config, in beta3 or earlier
            if(params != null)
                confFromJson.setDataType(params.dataType());
            MultiLayerNetwork network = new MultiLayerNetwork(confFromJson);
            network.init(params, false);

            if (gotUpdaterState && updaterState != null) {
                network.getUpdater().setStateViewArray(network, updaterState, false);
            }
            return new Pair<>(network, zipFile);
        } else
            throw new IllegalStateException("Model wasnt found within file: gotConfig: [" + gotConfig
                    + "], gotCoefficients: [" + gotCoefficients + "], gotUpdater: [" + gotUpdaterState + "]");
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
        is = new CloseShieldInputStream(is);

        Pair<MultiLayerNetwork,Map<String,byte[]>> p = restoreMultiLayerNetworkHelper(is, loadUpdater);
        MultiLayerNetwork net = p.getFirst();
        Normalizer norm = restoreNormalizerFromMap(p.getSecond());
        return new Pair<>(net, norm);
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
        try(InputStream is = new BufferedInputStream(new FileInputStream(file))){
            return restoreMultiLayerNetworkAndNormalizer(is, loadUpdater);
        }
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
        return restoreComputationGraphHelper(is, loadUpdater).getFirst();
    }

    private static Pair<ComputationGraph,Map<String,byte[]>> restoreComputationGraphHelper(@NonNull InputStream is, boolean loadUpdater)
            throws IOException {
        checkInputStream(is);

        Map<String, byte[]> files = loadZipData(is);

        boolean gotConfig = false;
        boolean gotCoefficients = false;
        boolean gotUpdaterState = false;
        boolean gotPreProcessor = false;

        String json = "";
        INDArray params = null;
        INDArray updaterState = null;
        DataSetPreProcessor preProcessor = null;


        byte[] config = files.get(CONFIGURATION_JSON);
        if (config != null) {
            //restoring configuration

            InputStream stream = new ByteArrayInputStream(config);
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


        byte[] coefficients = files.get(COEFFICIENTS_BIN);
        if (coefficients != null) {
            if(coefficients.length > 0) {
                InputStream stream = new ByteArrayInputStream(coefficients);
                DataInputStream dis = new DataInputStream(stream);
                params = Nd4j.read(dis);

                dis.close();
                gotCoefficients = true;
            } else {
                byte[] noParamsMarker = files.get(NO_PARAMS_MARKER);
                gotCoefficients = (noParamsMarker != null);
            }
        }


        if (loadUpdater) {
            byte[] updaterStateEntry = files.get(UPDATER_BIN);
            if (updaterStateEntry != null) {
                InputStream stream = new ByteArrayInputStream(updaterStateEntry);
                DataInputStream dis = new DataInputStream(stream);
                updaterState = Nd4j.read(dis);

                dis.close();
                gotUpdaterState = true;
            }
        }

        byte[] prep = files.get(PREPROCESSOR_BIN);
        if (prep != null) {
            InputStream stream = new ByteArrayInputStream(prep);
            ObjectInputStream ois = new ObjectInputStream(stream);

            try {
                preProcessor = (DataSetPreProcessor) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e);
            }

            gotPreProcessor = true;
        }


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

            //Handle legacy config - no network DataType in config, in beta3 or earlier
            if(params != null)
                confFromJson.setDataType(params.dataType());

            ComputationGraph cg = new ComputationGraph(confFromJson);
            cg.init(params, false);


            if (gotUpdaterState && updaterState != null) {
                cg.getUpdater().setStateViewArray(updaterState);
            }
            return new Pair<>(cg, files);
        } else
            throw new IllegalStateException("Model wasnt found within file: gotConfig: [" + gotConfig
                    + "], gotCoefficients: [" + gotCoefficients + "], gotUpdater: [" + gotUpdaterState + "]");
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


        Pair<ComputationGraph,Map<String,byte[]>> p = restoreComputationGraphHelper(is, loadUpdater);
        ComputationGraph net = p.getFirst();
        Normalizer norm = restoreNormalizerFromMap(p.getSecond());
        return new Pair<>(net, norm);
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
    	return restoreComputationGraphAndNormalizer(new FileInputStream(file), loadUpdater);
    }

    /**
     * Load a computation graph from a file
     * @param file the file to get the computation graph from
     * @return the loaded computation graph
     *
     * @throws IOException
     */
    public static ComputationGraph restoreComputationGraph(@NonNull File file, boolean loadUpdater) throws IOException {
    	return restoreComputationGraph(new FileInputStream(file), loadUpdater);
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
            tempFile = ND4JFileUtils.createTempFile("dl4jModelSerializerTemp", "bin");
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
            tempFile = ND4JFileUtils.createTempFile("dl4jModelSerializerTemp", "bin");
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

                try(ByteArrayOutputStream baos = new ByteArrayOutputStream(); ObjectOutputStream oos = new ObjectOutputStream(baos)){
                    oos.writeObject(o);
                    byte[] bytes = baos.toByteArray();
                    ZipEntry entry = new ZipEntry("objects/" + key);
                    entry.setSize(bytes.length);
                    writeFile.putNextEntry(entry);
                    writeFile.write(bytes);
                    writeFile.closeEntry();
                }

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
    public static <T extends Normalizer> T restoreNormalizerFromFile(File file) throws IOException {
        try (InputStream is = new BufferedInputStream(new FileInputStream(file))) {
        	return restoreNormalizerFromInputStream(is);
        } catch (Exception e) {
            log.warn("Error while restoring normalizer, trying to restore assuming deprecated format...");
            DataNormalization restoredDeprecated = restoreNormalizerFromInputStreamDeprecated(new FileInputStream(file));

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
        Map<String, byte[]> files = loadZipData(is);
        return restoreNormalizerFromMap(files);
    }

    private static <T extends Normalizer> T restoreNormalizerFromMap(Map<String, byte[]> files) throws IOException {
        byte[] norm = files.get(NORMALIZER_BIN);

        // checking for file existence
        if (norm == null)
            return null;
        try {
        	return NormalizerSerializer.getDefault().restore(new ByteArrayInputStream(norm));
        }
        catch (Exception e) {
        	throw new IOException("Error loading normalizer", e);
		}
    }

    /**
     * @deprecated
     *
     * This method restores normalizer from a given persisted model file serialized with Java object serialization
     *
     */
    private static DataNormalization restoreNormalizerFromInputStreamDeprecated(InputStream stream) {
    	try {
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
        //available method can return 0 in some cases: https://github.com/eclipse/deeplearning4j/issues/4887
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
    }

    private static Map<String, byte[]> loadZipData(InputStream is) throws IOException {
        return loadZipData(is, maxTotalUncompressedSize, maxCompressionRatio, maxZipEntries);
    }

    /**
     * Load ZIP data with explicit security limits.
     * @param is Input stream containing ZIP data
     * @param maxTotalSize Maximum total uncompressed size in bytes
     * @param maxRatio Maximum compression ratio allowed
     * @param maxEntries Maximum number of ZIP entries allowed
     * @return Map of entry names to their decompressed byte contents
     * @throws IOException If an I/O error occurs or security limits are exceeded
     */
    private static Map<String, byte[]> loadZipData(InputStream is, long maxTotalSize,
            double maxRatio, int maxEntries) throws IOException {
        Map<String, byte[]> result = new HashMap<>();
        long totalBytesRead = 0;
        int entryCount = 0;

        try (final ZipInputStream zis = new ZipInputStream(is)) {
            ZipEntry zipEntry;
            while ((zipEntry = zis.getNextEntry()) != null) {
                // Check entry count limit
                entryCount++;
                if (entryCount > maxEntries) {
                    throw new IOException("Potential zip bomb detected: too many entries. " +
                            "Found " + entryCount + " entries, maximum allowed is " + maxEntries + ". " +
                            "If this is a legitimate model file, increase the limit using " +
                            "ModelSerializer.setMaxZipEntries() or system property 'dl4j.model.maxZipEntries'");
                }

                if (zipEntry.isDirectory()) {
                    continue;
                }

                String entryName = zipEntry.getName();
                long compressedSize = zipEntry.getCompressedSize();
                long uncompressedSize = zipEntry.getSize();

                // Check compression ratio if both sizes are known
                if (compressedSize > 0 && uncompressedSize > 0) {
                    double ratio = (double) uncompressedSize / compressedSize;
                    if (ratio > maxRatio) {
                        throw new IOException("Potential zip bomb detected: suspicious compression ratio. " +
                                "Entry '" + entryName + "' has ratio " + String.format("%.1f", ratio) +
                                ":1 (compressed: " + compressedSize + " bytes, uncompressed: " + uncompressedSize + " bytes). " +
                                "Maximum allowed ratio is " + String.format("%.1f", maxRatio) + ":1. " +
                                "If this is a legitimate model file, increase the limit using " +
                                "ModelSerializer.setMaxCompressionRatio() or system property 'dl4j.model.maxCompressionRatio'");
                    }
                }

                // Check if claimed uncompressed size alone would exceed limit
                if (uncompressedSize > 0 && (totalBytesRead + uncompressedSize) > maxTotalSize) {
                    throw new IOException("Potential zip bomb detected: total uncompressed size would exceed limit. " +
                            "Entry '" + entryName + "' claims " + uncompressedSize + " bytes, " +
                            "which would bring total to " + (totalBytesRead + uncompressedSize) + " bytes. " +
                            "Maximum allowed is " + maxTotalSize + " bytes (" + (maxTotalSize / (1024 * 1024)) + " MB). " +
                            "If this is a legitimate model file, increase the limit using " +
                            "ModelSerializer.setMaxTotalUncompressedSize() or system property 'dl4j.model.maxZipSize'");
                }

                // Read entry data with actual byte counting (don't trust ZIP headers)
                byte[] data = readZipEntryWithLimit(zis, entryName, maxTotalSize - totalBytesRead);
                totalBytesRead += data.length;

                // Verify we haven't exceeded total limit (double-check after actual read)
                if (totalBytesRead > maxTotalSize) {
                    throw new IOException("Potential zip bomb detected: total uncompressed size exceeded limit. " +
                            "Read " + totalBytesRead + " bytes so far, maximum allowed is " + maxTotalSize + " bytes. " +
                            "If this is a legitimate model file, increase the limit using " +
                            "ModelSerializer.setMaxTotalUncompressedSize() or system property 'dl4j.model.maxZipSize'");
                }

                result.put(entryName, data);
            }
        }

        log.debug("Successfully loaded ZIP data: {} entries, {} total bytes", entryCount, totalBytesRead);
        return result;
    }

    /**
     * Read a ZIP entry with size limit protection.
     * This method reads data incrementally and tracks actual bytes read,
     * rather than trusting the ZIP header's claimed size.
     */
    private static byte[] readZipEntryWithLimit(ZipInputStream zis, String entryName, long remainingAllowed)
            throws IOException {
        ByteArrayOutputStream bout = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        long totalRead = 0;
        int bytesRead;

        while ((bytesRead = zis.read(buffer)) != -1) {
            totalRead += bytesRead;

            // Check limit during read, not just at the end
            if (totalRead > remainingAllowed) {
                throw new IOException("Potential zip bomb detected while reading entry '" + entryName + "'. " +
                        "Entry exceeded remaining size allowance of " + remainingAllowed + " bytes. " +
                        "If this is a legitimate model file, increase the limit using " +
                        "ModelSerializer.setMaxTotalUncompressedSize() or system property 'dl4j.model.maxZipSize'");
            }

            // Also check for unreasonably large individual entries
            if (totalRead > Integer.MAX_VALUE - 8) {
                throw new IOException("Entry '" + entryName + "' exceeds maximum supported size (2GB)");
            }

            bout.write(buffer, 0, bytesRead);
        }

        return bout.toByteArray();
    }

}
