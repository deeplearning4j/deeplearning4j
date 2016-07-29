package org.deeplearning4j.util;

import lombok.NonNull;
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
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * Utility class suited to save/restore neural net models
 *
 * @author raver119@gmail.com
 */
public class ModelSerializer {

    public static final String UPDATER_BIN = "updater.bin";

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
        ZipOutputStream zipfile = new ZipOutputStream(stream);

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
            ZipEntry updater = new ZipEntry(UPDATER_BIN);
            zipfile.putNextEntry(updater);


            bos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(bos);
            if (model instanceof  MultiLayerNetwork) {
                oos.writeObject(((MultiLayerNetwork) model).getUpdater());
            } else if (model instanceof ComputationGraph) {
                oos.writeObject(((ComputationGraph) model).getUpdater());
            }
            oos.flush();
            oos.close();

            inputStream = new ByteArrayInputStream(bos.toByteArray());
            writeEntry(inputStream, zipfile);
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
     * Load a multi layer network
     * from a file
     * @param file the file to load from
     * @return the loaded multi layer network
     * @throws IOException
     */
    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull File file) throws IOException {
        ZipFile zipFile = new ZipFile(file);

        boolean gotConfig = false;
        boolean gotCoefficients = false;
        boolean gotUpdater = false;
        boolean gotPreProcessor = false;

        String json = "";
        INDArray params = null;
        Updater updater = null;
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


        ZipEntry updaters = zipFile.getEntry(UPDATER_BIN);
        if (updaters != null) {
            InputStream stream = zipFile.getInputStream(updaters);
            ObjectInputStream ois = new ObjectInputStream(stream);

            try {
                updater = (Updater) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e);
            }

            gotUpdater = true;
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


            if (gotUpdater && updater != null) {
                network.setUpdater(updater);
            }
            return network;
        } else throw new IllegalStateException("Model wasnt found within file: gotConfig: ["+ gotConfig+"], gotCoefficients: ["+ gotCoefficients+"], gotUpdater: ["+gotUpdater+"]");
    }


    /**
     * Load a multi layer network
     * from a file
     * @param is the inputstream to load from
     * @return the loaded multi layer network
     * @throws IOException
     */
    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull InputStream is) throws IOException {
        ZipInputStream zipFile = new ZipInputStream(is);

        boolean gotConfig = false;
        boolean gotCoefficients = false;
        boolean gotUpdater = false;
        boolean gotPreProcessor = false;

        String json = "";
        INDArray params = null;
        DataSetPreProcessor preProcessor = null;
        Updater updater = null;


        ZipEntry entry;
        while((entry = zipFile.getNextEntry()) != null) {
            switch (entry.getName()) {
                case "configuration.json":
                    DataInputStream dis = new DataInputStream(zipFile);
                    params = Nd4j.read(dis);
                    gotConfig = true;
                    break;
                case "coefficients.bin":
                    DataInputStream dis2 = new DataInputStream(zipFile);
                    params = Nd4j.read(dis2);
                    gotCoefficients = true;
                    break;
                case UPDATER_BIN: {
                    ObjectInputStream ois = new ObjectInputStream(zipFile);

                    try {
                        updater = (Updater) ois.readObject();
                    } catch (ClassNotFoundException e) {
                        throw new RuntimeException(e);
                    }

                    gotUpdater = true;
                    }
                    break;
                case "preprocessor.bin": {
                    ObjectInputStream ois = new ObjectInputStream(zipFile);

                    try {
                        preProcessor = (DataSetPreProcessor) ois.readObject();
                    } catch (ClassNotFoundException e) {
                        throw new RuntimeException(e);
                    }

                    gotPreProcessor = true;
                }
                break;

            }

            zipFile.closeEntry();

        }


        zipFile.close();

        if (gotConfig && gotCoefficients) {
            MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(json);
            MultiLayerNetwork network = new MultiLayerNetwork(confFromJson);
            network.init(params, false);

            if (gotUpdater && updater != null) {
                network.setUpdater(updater);
            }
            return network;
        } else throw new IllegalStateException("Model wasnt found within file: gotConfig: ["+ gotConfig+"], gotCoefficients: ["+ gotCoefficients+"], gotUpdater: ["+gotUpdater+"]");
    }

    /**
     *
     * @param path
     * @return
     * @throws IOException
     */
    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull String path) throws IOException {
        return restoreMultiLayerNetwork(new File(path));
    }

    /**
     *
     * @param path
     * @return
     * @throws IOException
     */
    public static ComputationGraph restoreComputationGraph(@NonNull String path) throws IOException {
        return restoreComputationGraph(new File(path));
    }


    /**
     * Load a computation graph from a file
     * @param is the inputstream to get the computation graph from
     * @return the loaded computation graph
     *
     * @throws IOException
     */
    public static ComputationGraph restoreComputationGraph(@NonNull InputStream is) throws IOException {
        ZipInputStream zis = new ZipInputStream(is);
        boolean gotConfig = false;
        boolean gotCoefficients = false;
        boolean gotUpdater = false;
        boolean gotPreProcessor = false;

        String json = "";
        INDArray params = null;
        ComputationGraphUpdater updater = null;
        DataSetPreProcessor preProcessor = null;
        BufferedReader reader = new BufferedReader(new InputStreamReader(zis));

        ZipEntry entry;
        while((entry = zis.getNextEntry()) != null) {
            switch(entry.getName()) {
                case "configuration.json":
                    String line;
                    StringBuilder js = new StringBuilder();
                    while ((line = reader.readLine()) != null) {
                        js.append(line).append("\n");
                    }
                    json = js.toString();

                    gotConfig = true;
                    break;
                case "coefficients.bin":
                    DataInputStream dis = new DataInputStream(zis);
                    params = Nd4j.read(dis);

                    gotCoefficients = true;
                    break;
                case UPDATER_BIN:{
                    ObjectInputStream ois = new ObjectInputStream(zis);

                    try {
                        updater = (ComputationGraphUpdater) ois.readObject();
                    } catch (ClassNotFoundException e) {
                        throw new RuntimeException(e);
                    }

                    gotUpdater = true;
                    }
                    break;
                case "preprocessor.bin": {
                    ObjectInputStream ois = new ObjectInputStream(zis);

                    try {
                        preProcessor = (DataSetPreProcessor) ois.readObject();
                    } catch (ClassNotFoundException e) {
                        throw new RuntimeException(e);
                    }

                    gotPreProcessor = true;
                    }
                    break;
            }

            zis.closeEntry();
        }

        if (gotConfig && gotCoefficients) {
            ComputationGraphConfiguration confFromJson = ComputationGraphConfiguration.fromJson(json);
            ComputationGraph cg = new ComputationGraph(confFromJson);
            cg.init(params, false);

            if (gotUpdater && updater != null) {
                cg.setUpdater(updater);
            }

            zis.close();

            return cg;
        }
        else {
            zis.close();
            throw new IllegalStateException("Model wasnt found within file: gotConfig: [" + gotConfig + "], gotCoefficients: [" + gotCoefficients + "], gotUpdater: [" + gotUpdater + "]");
        }
    }

    /**
     * Load a computation graph from a file
     * @param file the file to get the computation graph from
     * @return the loaded computation graph
     *
     * @throws IOException
     */
    public static ComputationGraph restoreComputationGraph(@NonNull File file) throws IOException {
        ZipFile zipFile = new ZipFile(file);

        boolean gotConfig = false;
        boolean gotCoefficients = false;
        boolean gotUpdater = false;
        boolean gotPreProcessor = false;

        String json = "";
        INDArray params = null;
        ComputationGraphUpdater updater = null;
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


        ZipEntry updaters = zipFile.getEntry(UPDATER_BIN);
        if (updaters != null) {
            InputStream stream = zipFile.getInputStream(updaters);
            ObjectInputStream ois = new ObjectInputStream(stream);

            try {
                updater = (ComputationGraphUpdater) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e);
            }

            gotUpdater = true;
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


            if (gotUpdater && updater != null) {
                cg.setUpdater(updater);
            }
            return cg;
        }
        else throw new IllegalStateException("Model wasnt found within file: gotConfig: [" + gotConfig + "], gotCoefficients: [" + gotCoefficients + "], gotUpdater: [" + gotUpdater+  "]");
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
