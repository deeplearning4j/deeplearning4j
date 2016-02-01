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

package org.deeplearning4j.earlystopping.saver;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;

/** Save the best (and latest/most recent) {@link ComputationGraph}s learned during early stopping training to the local file system.<br>
 * Instances of this class will save 3 files for best (and optionally, latest) models:<br>
 * (a) The network configuration: bestGraphConf.json<br>
 * (b) The network parameters: bestGraphParams.bin<br>
 * (c) The network updater: bestGraphUpdater.bin<br>
 * <br>
 * NOTE: The model updater is an object that contains the internal state for training features such as AdaGrad, Momentum
 * and RMSProp.<br>
 * The updater is <i>not</i> required to use the network at test time; it is saved in case further training is required.
 * Without saving the updater, any further training would result in the updater being recreated, without the benefit
 * of the history/internal state. This could negatively impact training performance after loading the network.
 *
 * @author Alex Black
 */
public class LocalFileGraphSaver implements EarlyStoppingModelSaver<ComputationGraph> {

    private static final String bestFileNameConf = "bestGraphConf.json";
    private static final String bestFileNameParam = "bestGraphParams.bin";
    private static final String bestFileNameUpdater = "bestGraphUpdater.bin";
    private static final String latestFileNameConf = "latestGraphConf.json";
    private static final String latestFileNameParam = "latestGraphParams.bin";
    private static final String latestFileNameUpdater = "latestGraphUpdater.bin";

    private String directory;
    private Charset encoding;

    /**Constructor that uses default character set for configuration (json) encoding
     * @param directory Directory to save networks
     */
    public LocalFileGraphSaver(String directory) {
        this(directory, Charset.defaultCharset());
    }

    /**
     * @param directory Directory to save networks
     * @param encoding Character encoding for configuration (json)
     */
    public LocalFileGraphSaver(String directory, Charset encoding){
        this.directory = directory;
        this.encoding = encoding;
    }

    @Override
    public void saveBestModel(ComputationGraph net, double score) throws IOException {
        String confOut = FilenameUtils.concat(directory,bestFileNameConf);
        String paramOut = FilenameUtils.concat(directory,bestFileNameParam);
        String updaterOut = FilenameUtils.concat(directory,bestFileNameUpdater);
        save(net,confOut,paramOut,updaterOut);
    }

    @Override
    public void saveLatestModel(ComputationGraph net, double score) throws IOException {
        String confOut = FilenameUtils.concat(directory,latestFileNameConf);
        String paramOut = FilenameUtils.concat(directory,latestFileNameParam);
        String updaterOut = FilenameUtils.concat(directory,latestFileNameUpdater);
        save(net,confOut,paramOut,updaterOut);
    }

    private void save(ComputationGraph net, String confOut, String paramOut, String updaterOut) throws IOException{
        String confJSON = net.getConfiguration().toJson();
        INDArray params = net.params();
        ComputationGraphUpdater updater = net.getUpdater();

        FileUtils.writeStringToFile(new File(confOut), confJSON, encoding);
        try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(paramOut)))){
            Nd4j.write(params, dos);
        }

        try(ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(updaterOut)))){
            oos.writeObject(updater);
        }
    }

    @Override
    public ComputationGraph getBestModel() throws IOException {
        String confOut = FilenameUtils.concat(directory, bestFileNameConf);
        String paramOut = FilenameUtils.concat(directory, bestFileNameParam);
        String updaterOut = FilenameUtils.concat(directory, bestFileNameUpdater);
        return load(confOut, paramOut, updaterOut);
    }

    @Override
    public ComputationGraph getLatestModel() throws IOException {
        String confOut = FilenameUtils.concat(directory,bestFileNameConf);
        String paramOut = FilenameUtils.concat(directory,bestFileNameParam);
        String updaterOut = FilenameUtils.concat(directory,bestFileNameUpdater);
        return load(confOut,paramOut,updaterOut);
    }

    private ComputationGraph load(String confOut, String paramOut, String updaterOut) throws IOException {
        String confJSON = FileUtils.readFileToString(new File(confOut), encoding);
        INDArray params;
        ComputationGraphUpdater updater;
        try(DataInputStream dis = new DataInputStream(Files.newInputStream(Paths.get(paramOut)))){
            params = Nd4j.read(dis);
        }
        try(ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(updaterOut)))){
            updater = (ComputationGraphUpdater)ois.readObject();
        }catch(ClassNotFoundException e){
            throw new RuntimeException(e);  //Should never happen
        }
        ComputationGraphConfiguration conf = ComputationGraphConfiguration.fromJson(confJSON);
        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setParams(params);
        net.setUpdater(updater);
        return net;
    }

    @Override
    public String toString(){
        return "LocalFileGraphSaver(dir=" + directory + ")";
    }
}
