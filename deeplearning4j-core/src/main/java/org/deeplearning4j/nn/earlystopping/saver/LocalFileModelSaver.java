package org.deeplearning4j.nn.earlystopping.saver;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;

public class LocalFileModelSaver implements EarlyStoppingModelSaver {

    private static final String bestFileNameConf = "bestModelConf.json";
    private static final String bestFileNameParam = "bestModelParams.bin";
    private static final String bestFileNameUpdater = "bestModelUpdater.bin";
    private static final String latestFileNameConf = "latestModelConf.json";
    private static final String latestFileNameParam = "latestModelParams.bin";
    private static final String latestFileNameUpdater = "latestModelUpdater.bin";

    private String directory;
    private Charset encoding;

    public LocalFileModelSaver(String directory) {
        this(directory, Charset.defaultCharset());
    }

    public LocalFileModelSaver(String directory, Charset encoding){
        this.directory = directory;
        this.encoding = encoding;
    }

    @Override
    public void saveBestModel(MultiLayerNetwork net, double score) throws IOException {
        String confOut = FilenameUtils.concat(directory,bestFileNameConf);
        String paramOut = FilenameUtils.concat(directory,bestFileNameParam);
        String updaterOut = FilenameUtils.concat(directory,bestFileNameUpdater);
        save(net,confOut,paramOut,updaterOut);
    }

    @Override
    public void saveLatestModel(MultiLayerNetwork net, double score) throws IOException {
        String confOut = FilenameUtils.concat(directory,latestFileNameConf);
        String paramOut = FilenameUtils.concat(directory,latestFileNameParam);
        String updaterOut = FilenameUtils.concat(directory,latestFileNameUpdater);
        save(net,confOut,paramOut,updaterOut);
    }

    private void save(MultiLayerNetwork net, String confOut, String paramOut, String updaterOut) throws IOException{
        String confJSON = net.getLayerWiseConfigurations().toJson();
        INDArray params = net.params();
        Updater updater = net.getUpdater();

        FileUtils.writeStringToFile(new File(confOut),confJSON,encoding);
        try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(paramOut)))){
            Nd4j.write(params, dos);
        }

        try(ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(updaterOut)))){
            oos.writeObject(updater);
        }
    }

    @Override
    public MultiLayerNetwork getBestModel() throws IOException {
        String confOut = FilenameUtils.concat(directory,bestFileNameConf);
        String paramOut = FilenameUtils.concat(directory,bestFileNameParam);
        String updaterOut = FilenameUtils.concat(directory,bestFileNameUpdater);
        return load(confOut, paramOut, updaterOut);
    }

    @Override
    public MultiLayerNetwork getLatestModel() throws IOException {
        String confOut = FilenameUtils.concat(directory,bestFileNameConf);
        String paramOut = FilenameUtils.concat(directory,bestFileNameParam);
        String updaterOut = FilenameUtils.concat(directory,bestFileNameUpdater);
        return load(confOut,paramOut,updaterOut);
    }

    private MultiLayerNetwork load(String confOut, String paramOut, String updaterOut) throws IOException {
        String confJSON = FileUtils.readFileToString(new File(confOut),encoding);
        INDArray params;
        Updater updater;
        try(DataInputStream dis = new DataInputStream(Files.newInputStream(Paths.get(paramOut)))){
            params = Nd4j.read(dis);
        }
        try(ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(updaterOut)))){
            updater = (Updater)ois.readObject();
        }catch(ClassNotFoundException e){
            throw new RuntimeException(e);  //Should never happen
        }
        MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(confJSON);
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setParams(params);
        net.setUpdater(updater);
        return net;
    }
}
