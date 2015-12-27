package org.arbiter.deeplearning4j.saver.local;

import lombok.AllArgsConstructor;
import org.apache.commons.io.FileUtils;
import org.arbiter.deeplearning4j.DL4JConfiguration;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

@AllArgsConstructor
public class LocalFileMultiLayerNetworkResultReference<A> implements ResultReference<DL4JConfiguration,MultiLayerNetwork,A> {

    private int index;
    private String dir;
    private File configFile;
    private File networkParamsFile;
    private File scoreFile;
    private File additionalResultsFile;
    private File esConfigFile;
    private File numEpochsFile;
    private Candidate<DL4JConfiguration> candidate;

    @Override
    public OptimizationResult<DL4JConfiguration, MultiLayerNetwork,A> getResult() throws IOException {
        String jsonConfig = FileUtils.readFileToString(configFile);
        INDArray params;
        try( DataInputStream dis = new DataInputStream(new FileInputStream(networkParamsFile)) ){
            params = Nd4j.read(dis);
        }

        MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(jsonConfig);
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setParams(params);

        String scoreStr = FileUtils.readFileToString(scoreFile);
        //TODO: properly parsing. Probably want to store additional info other than just score...
        double d = Double.parseDouble(scoreStr);

        EarlyStoppingConfiguration earlyStoppingConfiguration = null;
        if(esConfigFile != null && esConfigFile.exists()){
            try(ObjectInputStream ois = new ObjectInputStream(new FileInputStream(esConfigFile))){
                earlyStoppingConfiguration = (EarlyStoppingConfiguration)ois.readObject();
            } catch( ClassNotFoundException e){
                throw new RuntimeException("Error loading early stopping configuration",e);
            }
        }
        int nEpochs = 1;
        if(numEpochsFile != null && numEpochsFile.exists()){
            String numEpochs = FileUtils.readFileToString(numEpochsFile);
            nEpochs = Integer.parseInt(numEpochs);
        }

        DL4JConfiguration dl4JConfiguration = new DL4JConfiguration(conf,earlyStoppingConfiguration,nEpochs);



        A additionalResults;
        if(additionalResultsFile.exists()) {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(additionalResultsFile))) {
                additionalResults = (A) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException("Error loading additional results",e);
            }
        } else {
            additionalResults = null;
        }

        return new OptimizationResult<>(candidate,net,d,index,additionalResults);
    }

    @Override
    public String toString(){
        return "LocalFileMultiLayerNetworkResultReference(" + dir + ")";
    }
}
