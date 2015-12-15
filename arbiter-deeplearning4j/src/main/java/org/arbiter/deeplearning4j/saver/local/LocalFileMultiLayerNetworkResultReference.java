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

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

@AllArgsConstructor
public class LocalFileMultiLayerNetworkResultReference implements ResultReference<MultiLayerConfiguration,MultiLayerNetwork> {

    private File configFile;
    private File networkParamsFile;
    private File scoreFile;
    private Candidate<MultiLayerConfiguration> candidate;

    @Override
    public OptimizationResult<MultiLayerConfiguration, MultiLayerNetwork> getResult() throws IOException {
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

        return new OptimizationResult<>(candidate,net,d,-1);     //TODO index
    }
}
