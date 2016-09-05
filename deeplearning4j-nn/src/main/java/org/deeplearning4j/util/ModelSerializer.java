package org.deeplearning4j.util;

import lombok.NonNull;
import org.apache.commons.io.output.CloseShieldOutputStream;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.reports.Task;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * Utility class suited to save/restore neural net models
 *
 * @author raver119@gmail.com
 */
public class ModelSerializer {

    public static final String OLD_UPDATER_BIN = "updater.bin";
    public static final String UPDATER_BIN = "updaterState.bin";

    private ModelSerializer() {
    }

    /**
     * Write a model to a file
     * @param model the model to write
     * @param file the file to write to
     * @param saveUpdater whether to save the updater or not
     * @throws IOException
     */
    public static void writeModel(@NonNull Model model, @NonNull File file, boolean saveUpdater) throws IOException {
        try(BufferedOutputStream stream = new BufferedOutputStream(new FileOutputStream(file))){
            writeModel(model, stream, saveUpdater);
            stream.flush();
            stream.close();
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
        try(BufferedOutputStream stream = new BufferedOutputStream(new FileOutputStream(path))){
            writeModel(model, stream, saveUpdater);
            stream.flush();
            stream.close();
        }
    }

    /**
     * Write a model to an output stream
     * @param model the model to save
     * @param stream the output stream to write to
     * @param saveUpdater whether to save the updater for the model or not
     * @throws IOException
     */
    public static void writeModel(@NonNull Model model, @NonNull OutputStream stream, boolean saveUpdater) throws IOException {
        ZipOutputStream zipfile = new ZipOutputStream(new CloseShieldOutputStream(stream));

        // save json first
        String json = "";
        if (model instanceof MultiLayerNetwork) {
            json = ((MultiLayerNetwork) model).getLayerWiseConfigurations().toJson();
        } else if (model instanceof ComputationGraph) {
            json = ((ComputationGraph) model).getConfiguration().toJson();
        }

        ZipEntry config = new ZipEntry("configuration.json");
        zipfile.putNextEntry(config);

        writeEntry(new ByteArrayInputStream(json.getBytes()), zipfile);

        ZipEntry coefficients = new ZipEntry("coefficients.bin");
        zipfile.putNextEntry(coefficients);

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        Nd4j.write(model.params(), dos);
        dos.flush();
        dos.close();

        InputStream inputStream = new ByteArrayInputStream(bos.toByteArray());
        writeEntry(inputStream, zipfile);

        if (saveUpdater) {
            INDArray updaterState = null;
            if (model instanceof  MultiLayerNetwork) {
                updaterState = ((MultiLayerNetwork) model).getUpdater().getStateViewArray();
            } else if (model instanceof ComputationGraph) {
                updaterState = ((ComputationGraph) model).getUpdater().getStateViewArray();
            }

            if(updaterState != null && updaterState.length() > 0){
                ZipEntry updater = new ZipEntry(UPDATER_BIN);
                zipfile.putNextEntry(updater);

                bos = new ByteArrayOutputStream();
                dos = new DataOutputStream(bos);
                Nd4j.write(updaterState, dos);
                dos.flush();
                dos.close();

                inputStream = new ByteArrayInputStream(bos.toByteArray());
                writeEntry(inputStream, zipfile);
            }
        }

        zipfile.flush();
        zipfile.close();
    }


    private static void writeEntry(InputStream inputStream, ZipOutputStream zipStream) throws IOException {
        byte[] bytes = new byte[1024];
        int bytesRead;
        while ((bytesRead = inputStream.read(bytes)) != -1) {
            zipStream.write(bytes, 0, bytesRead);
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
    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull File file, boolean loadUpdater) throws IOException {
        ZipFile zipFile = new ZipFile(file);

        boolean gotConfig = false;
        boolean gotCoefficients = false;
        boolean gotOldUpdater = false;
        boolean gotUpdaterState = false;
        boolean gotPreProcessor = false;

        String json = "";
        INDArray params = null;
        Updater updater = null;
        INDArray updaterState = null;
        DataSetPreProcessor preProcessor = null;


        ZipEntry config = zipFile.getEntry("configuration.json");
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


        ZipEntry coefficients = zipFile.getEntry("coefficients.bin");
        if (coefficients != null) {
            InputStream stream = zipFile.getInputStream(coefficients);
            DataInputStream dis = new DataInputStream(new BufferedInputStream(stream));
            params = Nd4j.read(dis);

            dis.close();
            gotCoefficients = true;
        }

        if (loadUpdater) {
            //This can be removed a few releases after 0.4.1...
            ZipEntry oldUpdaters = zipFile.getEntry(OLD_UPDATER_BIN);
            if (oldUpdaters != null) {
                InputStream stream = zipFile.getInputStream(oldUpdaters);
                ObjectInputStream ois = new ObjectInputStream(stream);

                try {
                    updater = (Updater) ois.readObject();
                } catch (ClassNotFoundException e) {
                    throw new RuntimeException(e);
                }

                gotOldUpdater = true;
            }

            ZipEntry updaterStateEntry = zipFile.getEntry(UPDATER_BIN);
            if (updaterStateEntry != null) {
                InputStream stream = zipFile.getInputStream(updaterStateEntry);
                DataInputStream dis = new DataInputStream(stream);
                updaterState = Nd4j.read(dis);

                dis.close();
                gotUpdaterState = true;
            }
        }

        ZipEntry prep = zipFile.getEntry("preprocessor.bin");
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
            MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(json);
            MultiLayerNetwork network = new MultiLayerNetwork(confFromJson);
            network.init(params, false);

            if(gotUpdaterState && updaterState != null){
                network.getUpdater().setStateViewArray(network, updaterState, false);
            } else if (gotOldUpdater && updater != null) {
                network.setUpdater(updater);
            }
            return network;
        } else throw new IllegalStateException("Model wasnt found within file: gotConfig: ["+ gotConfig+"], gotCoefficients: ["+ gotCoefficients+"], gotUpdater: ["+gotUpdaterState+"]");
    }


    /**
     * Load a MultiLayerNetwork from InputStream from a file
     *
     * @param is the inputstream to load from
     * @return the loaded multi layer network
     * @throws IOException
     */
    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull InputStream is, boolean loadUpdater) throws IOException {
        File tmpFile = File.createTempFile("restore", "multiLayer");
        Files.copy(is, Paths.get(tmpFile.getAbsolutePath()), StandardCopyOption.REPLACE_EXISTING);
        return restoreMultiLayerNetwork(tmpFile, loadUpdater);
    }

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
    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull String path, boolean loadUpdater) throws IOException {
        return restoreMultiLayerNetwork(new File(path), loadUpdater);
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
    public static ComputationGraph restoreComputationGraph(@NonNull String path, boolean loadUpdater) throws IOException {
        return restoreComputationGraph(new File(path), loadUpdater);
    }


    /**
     * Load a computation graph from a InputStream
     * @param is the inputstream to get the computation graph from
     * @return the loaded computation graph
     *
     * @throws IOException
     */
    public static ComputationGraph restoreComputationGraph(@NonNull InputStream is, boolean loadUpdater) throws IOException {
        File tmpFile = File.createTempFile("restore", "compGraph");
        Files.copy(is, Paths.get(tmpFile.getAbsolutePath()), StandardCopyOption.REPLACE_EXISTING);
        return restoreComputationGraph(tmpFile, loadUpdater);
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
        boolean gotOldUpdater = false;
        boolean gotUpdaterState = false;
        boolean gotPreProcessor = false;

        String json = "";
        INDArray params = null;
        ComputationGraphUpdater updater = null;
        INDArray updaterState = null;
        DataSetPreProcessor preProcessor = null;


        ZipEntry config = zipFile.getEntry("configuration.json");
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


        ZipEntry coefficients = zipFile.getEntry("coefficients.bin");
        if (coefficients != null) {
            InputStream stream = zipFile.getInputStream(coefficients);
            DataInputStream dis = new DataInputStream(stream);
            params = Nd4j.read(dis);

            dis.close();
            gotCoefficients = true;
        }


        if (loadUpdater) {
            ZipEntry oldUpdaters = zipFile.getEntry(OLD_UPDATER_BIN);
            if (oldUpdaters != null) {
                InputStream stream = zipFile.getInputStream(oldUpdaters);
                ObjectInputStream ois = new ObjectInputStream(stream);

                try {
                    updater = (ComputationGraphUpdater) ois.readObject();
                } catch (ClassNotFoundException e) {
                    throw new RuntimeException(e);
                }

                gotOldUpdater = true;
            }

            ZipEntry updaterStateEntry = zipFile.getEntry(UPDATER_BIN);
            if (updaterStateEntry != null) {
                InputStream stream = zipFile.getInputStream(updaterStateEntry);
                DataInputStream dis = new DataInputStream(stream);
                updaterState = Nd4j.read(dis);

                dis.close();
                gotUpdaterState = true;
            }
        }

        ZipEntry prep = zipFile.getEntry("preprocessor.bin");
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
            ComputationGraphConfiguration confFromJson = ComputationGraphConfiguration.fromJson(json);
            ComputationGraph cg = new ComputationGraph(confFromJson);
            cg.init(params, false);


            if(gotUpdaterState && updaterState != null){
                cg.getUpdater().setStateViewArray(updaterState);
            } else if (gotOldUpdater && updater != null) {
                cg.setUpdater(updater);
            }
            return cg;
        }
        else throw new IllegalStateException("Model wasnt found within file: gotConfig: [" + gotConfig + "], gotCoefficients: [" + gotCoefficients + "], gotUpdater: [" + gotUpdaterState+  "]");
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
                            if (layer instanceof RBM || layer instanceof org.deeplearning4j.nn.layers.feedforward.rbm.RBM) {
                                task.setArchitectureType(Task.ArchitectureType.RBM);
                                break;
                            }
                            if (layer.type().equals(Layer.Type.CONVOLUTIONAL)) {
                                task.setArchitectureType(Task.ArchitectureType.CONVOLUTION);
                                break;
                            } else if (layer.type().equals(Layer.Type.RECURRENT) || layer.type().equals(Layer.Type.RECURSIVE)) {
                                task.setArchitectureType(Task.ArchitectureType.RECURRENT);
                                break;
                            }
                        }
                    } else task.setArchitectureType(Task.ArchitectureType.UNKNOWN);
                } catch (Exception e) {
                    // do nothing here
                }
            } else if (model instanceof MultiLayerNetwork) {
                task.setNetworkType(Task.NetworkType.MultilayerNetwork);
                MultiLayerNetwork network = (MultiLayerNetwork) model;
                try {
                    if (network.getLayers() != null && network.getLayers().length > 0) {
                        for (Layer layer : network.getLayers()) {
                            if (layer instanceof RBM || layer instanceof org.deeplearning4j.nn.layers.feedforward.rbm.RBM) {
                                task.setArchitectureType(Task.ArchitectureType.RBM);
                                break;
                            }
                            if (layer.type().equals(Layer.Type.CONVOLUTIONAL)) {
                                task.setArchitectureType(Task.ArchitectureType.CONVOLUTION);
                                break;
                            } else if (layer.type().equals(Layer.Type.RECURRENT) || layer.type().equals(Layer.Type.RECURSIVE)) {
                                task.setArchitectureType(Task.ArchitectureType.RECURRENT);
                                break;
                            }
                        }
                    } else task.setArchitectureType(Task.ArchitectureType.UNKNOWN);
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
}
