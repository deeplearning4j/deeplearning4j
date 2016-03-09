package org.deeplearning4j.util;

import lombok.NonNull;
import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.io.FileUtils;
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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.reports.Task;
import org.nd4j.linalg.util.*;
import org.nd4j.linalg.util.SerializationUtils;

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

    public static void writeModel(@NonNull Model model, @NonNull File file, boolean saveUpdater) throws IOException {
        writeModel(model, new FileOutputStream(file), saveUpdater);
    }

    public static void writeModel(@NonNull Model model, @NonNull String path, boolean saveUpdater) throws IOException {
        writeModel(model, new FileOutputStream(path), saveUpdater);
    }

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
        org.nd4j.linalg.util.SerializationUtils.writeObject(model.params(),dos);
        dos.flush();
        dos.close();

        InputStream inputStream = new ByteArrayInputStream(bos.toByteArray());
        writeEntry(inputStream, zipfile);

        if (saveUpdater) {
            ZipEntry updater = new ZipEntry("updater.bin");
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

    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull File file) throws IOException {
        ZipFile zipFile = new ZipFile(file);

        boolean gotConfig = false;
        boolean gotCoefficients = false;
        boolean gotUpdater = false;

        String json = "";
        INDArray params = null;
        Updater updater = null;


        ZipEntry config = zipFile.getEntry("configuration.json");
        if (config != null) {
            //restoring configuration

            InputStream stream = zipFile.getInputStream(config);
            json = new String(IOUtils.toByteArray(stream));
            stream.close();
            gotConfig = true;
        }


        ZipEntry coefficients = zipFile.getEntry("coefficients.bin");
        if (coefficients != null) {
            InputStream stream = zipFile.getInputStream(coefficients);
            DataInputStream dis = new DataInputStream(stream);
            params = SerializationUtils.readObject(stream);

            dis.close();
            gotCoefficients = true;
        }


        ZipEntry updaters = zipFile.getEntry("updater.bin");
        if (updaters != null) {
            InputStream stream = zipFile.getInputStream(updaters);
            updater = SerializationUtils.readObject(stream);
            stream.close();
            gotUpdater = true;
        }


        zipFile.close();

        if (gotConfig && gotCoefficients) {
            MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(json);
            MultiLayerNetwork network = new MultiLayerNetwork(confFromJson);
            network.init();
            network.setParameters(params);


            if (gotUpdater && updater != null) {
                network.setUpdater(updater);
            }
            return network;
        } else throw new IllegalStateException("Model wasnt found within file: gotConfig: ["+ gotConfig+"], gotCoefficients: ["+ gotCoefficients+"], gotUpdater: ["+gotUpdater+"]");
    }

    public static MultiLayerNetwork restoreMultiLayerNetwork(@NonNull String path) throws IOException {
        return restoreMultiLayerNetwork(new File(path));
    }

    public static ComputationGraph restoreComputationGraph(@NonNull String path) throws IOException {
        return restoreComputationGraph(new File(path));
    }

    public static ComputationGraph restoreComputationGraph(@NonNull File file) throws IOException {
        ZipFile zipFile = new ZipFile(file);

        boolean gotConfig = false;
        boolean gotCoefficients = false;
        boolean gotUpdater = false;

        String json = "";
        INDArray params = null;
        ComputationGraphUpdater updater = null;


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
            params = SerializationUtils.readObject(dis);

            dis.close();
            gotCoefficients = true;
        }


        ZipEntry updaters = zipFile.getEntry("updater.bin");
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


        zipFile.close();

        if (gotConfig && gotCoefficients) {
            ComputationGraphConfiguration confFromJson = ComputationGraphConfiguration.fromJson(json);
            ComputationGraph cg = new ComputationGraph(confFromJson);
            cg.init();
            cg.setParams(params);


            if (gotUpdater && updater != null) {
                cg.setUpdater(updater);
            }
            return cg;
        } else throw new IllegalStateException("Model wasnt found within file: gotConfig: ["+ gotConfig+"], gotCoefficients: ["+ gotCoefficients+"], gotUpdater: ["+gotUpdater+"]");
    }

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
                    ; // do nothing here
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
                    ; // do nothing here
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
