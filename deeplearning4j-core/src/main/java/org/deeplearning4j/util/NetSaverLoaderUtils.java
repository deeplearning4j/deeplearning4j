package org.deeplearning4j.util;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Utility to save and load network configuration and parameters.
 */

public class NetSaverLoaderUtils {
    private static final Logger log = LoggerFactory.getLogger(NetSaverLoaderUtils.class);

    private NetSaverLoaderUtils(){}

    /**
     * Save model configuration and parameters
     * @param net trained network | model
     * @param basePath path to store configuration
     */
    public static void saveNetworkAndParameters(MultiLayerNetwork net, String basePath) {
        String confPath = FilenameUtils.concat(basePath, net.toString()+"-conf.json");
        String paramPath = FilenameUtils.concat(basePath, net.toString() + ".bin");
        log.info("Saving model and parameters to {} and {} ...",  confPath, paramPath);

        // save parameters
        try(DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(paramPath)))) {
            Nd4j.write(net.params(), dos);
            dos.flush();

            // save model configuration
            FileUtils.write(new File(confPath), net.conf().toJson());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Load existing model configuration and parameters
     * @param confPath string path where model configuration is stored
     * @param paramPath string path where parameters are stored
     */
    public static MultiLayerNetwork loadNetworkAndParameters(String confPath, String paramPath) {
        log.info("Loading saved model and parameters...");
        MultiLayerNetwork savedNetwork = null;
        // load parameters
        try {
            MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(confPath);
            DataInputStream dis = new DataInputStream(new FileInputStream(paramPath));
            INDArray newParams = Nd4j.read(dis);
            dis.close();

            // load model configuration
            savedNetwork = new MultiLayerNetwork(confFromJson);
            savedNetwork.init();
            savedNetwork.setParams(newParams);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return savedNetwork;
    }

    /**
     * Save existing parameters for the layer
     * @param param layer parameters in INDArray format
     * @param paramPath string path where parameters are stored
     */
    public static void saveLayerParameters(INDArray param, String paramPath)  {
        // save parameters for each layer
        log.info("Saving parameters to {} ...", paramPath);

        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(paramPath)))){
            Nd4j.write(param, dos);
            dos.flush();
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Load existing parameters to the layer
     * @param layer to load the parameters into
     * @param paramPath string path where parameters are stored
     */
    public static Layer loadLayerParameters(Layer layer, String paramPath) {
        // load parameters for each layer
        String name = layer.conf().getLayer().getLayerName();
        log.info("Loading saved parameters for layer {} ...", name);

        try{
            DataInputStream dis = new DataInputStream(new FileInputStream(paramPath));
            INDArray param = Nd4j.read(dis);
            dis.close();
            layer.setParams(param);
        } catch(IOException e) {
            e.printStackTrace();
        }

        return layer;
    }


    /**
     * Save existing parameters for the network
     * @param net trained network | model
     * @param layerIds list of *int* layer ids
     * @param paramPaths map of layer ids and string paths to store parameters
     */
    public static void saveParameters(MultiLayerNetwork net, int[] layerIds, Map<Integer, String> paramPaths) {
        Layer layer;
        for(int layerId: layerIds) {
            layer = net.getLayer(layerId);
            if (!layer.paramTable().isEmpty()) {
                NetSaverLoaderUtils.saveLayerParameters(layer.params(), paramPaths.get(layerId));
            }
        }
    }

    /**
     * Save existing parameters for the network
     * @param net trained network | model
     * @param layerIds list of *string* layer ids
     * @param paramPaths map of layer ids and string paths to store parameters
     */
    public static void saveParameters(MultiLayerNetwork net, String[] layerIds, Map<String, String> paramPaths) {
        Layer layer;
        for(String layerId: layerIds) {
            layer = net.getLayer(layerId);
            if (!layer.paramTable().isEmpty()) {
                NetSaverLoaderUtils.saveLayerParameters(layer.params(), paramPaths.get(layerId));
            }
        }
    }

    /**
     * Load existing parameters for the network
     * @param net trained network | model
     * @param layerIds list of *int* layer ids
     * @param paramPaths map of layer ids and string paths to find parameters
     */
    public static MultiLayerNetwork loadParameters(MultiLayerNetwork net, int[] layerIds, Map<Integer, String> paramPaths) {
        Layer layer;
        for(int layerId: layerIds) {
            layer = net.getLayer(layerId);
            loadLayerParameters(layer, paramPaths.get(layerId));
        }
        return net;
    }

    /**
     * Load existing parameters for the network
     * @param net trained network | model
     * @param layerIds list of *string* layer ids
     * @param paramPaths map of layer ids and string paths to find parameters
     */
    public static MultiLayerNetwork loadParameters(MultiLayerNetwork net, String[] layerIds, Map<String, String> paramPaths) {
        Layer layer;
        for(String layerId: layerIds) {
            layer = net.getLayer(layerId);
            loadLayerParameters(layer, paramPaths.get(layerId));
        }
        return net;
    }


    /**
     * Create map of *int* layerIds to path
     * @param layerIds list of *string* layer ids
     * @param basePath string path to find parameters
     */
    public static  Map<Integer, String> getIdParamPaths(String basePath, int[] layerIds){
        Map<Integer, String> paramPaths = new HashMap<>();
        for (int id : layerIds) {
            paramPaths.put(id, FilenameUtils.concat(basePath, id + ".bin"));
        }

        return paramPaths;
    }

    /**
     * Create map of *string* layerIds to path
     * @param layerIds list of *string* layer ids
     * @param basePath string path to find parameters
     */
    public static Map<String, String> getStringParamPaths(String basePath, String[] layerIds){
        Map<String, String> paramPaths = new HashMap<>();

        for (String name : layerIds) {
            paramPaths.put(name, FilenameUtils.concat(basePath, name + ".bin"));
        }

        return paramPaths;
    }

    /**
     * Define output directory based on network type
     * @param networkType
     */
    public static String defineOutputDir(String networkType){
        String tmpDir = System.getProperty("java.io.tmpdir");
        String outputPath = File.separator + networkType + File.separator + "output";
        File dataDir = new File(tmpDir,outputPath);
        if (!dataDir.getParentFile().exists())
            dataDir.mkdirs();
        return dataDir.toString();

    }

}
