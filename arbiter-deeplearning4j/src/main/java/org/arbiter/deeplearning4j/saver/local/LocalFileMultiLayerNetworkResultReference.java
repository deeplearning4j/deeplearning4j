package org.arbiter.deeplearning4j.saver.local;

import lombok.AllArgsConstructor;
import org.apache.commons.io.FileUtils;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

@AllArgsConstructor
public class LocalFileMultiLayerNetworkResultReference<A> implements ResultReference<MultiLayerConfiguration,MultiLayerNetwork,A> {

    private File configFile;
    private File networkParamsFile;
    private File scoreFile;
    private File additionalResultsFile;
    private Candidate<MultiLayerConfiguration> candidate;

    @Override
    public OptimizationResult<MultiLayerConfiguration, MultiLayerNetwork,A> getResult() throws IOException {
        String jsonConfig = FileUtils.readFileToString(configFile);
        INDArray params;
        try( DataInputStream dis = new DataInputStream(new FileInputStream(networkParamsFile)) ){
            params = Nd4j.read(dis);
        }
        String scoreStr = FileUtils.readFileToString(scoreFile);
        //TODO: properly parsing. Probably want to store additional info other than just score...
        double d = Double.parseDouble(scoreStr);

        MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(jsonConfig);
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setParams(params);

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

        return new OptimizationResult<>(candidate,net,d,-1,additionalResults);     //TODO index
    }
}
