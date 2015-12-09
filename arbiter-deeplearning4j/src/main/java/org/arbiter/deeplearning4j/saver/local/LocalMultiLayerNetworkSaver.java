package org.arbiter.deeplearning4j.saver.local;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

/**Basic MultiLayerNetwork saver. Saves config, parameters and score to: baseDir/0/, baseDir/1/, etc
 * where index is given by OptimizationResult.getIndex()
 */
public class LocalMultiLayerNetworkSaver implements ResultSaver<MultiLayerConfiguration,MultiLayerNetwork> {
    private static Logger log = LoggerFactory.getLogger(LocalMultiLayerNetworkSaver.class);
    private String path;

    public LocalMultiLayerNetworkSaver(String path){
        if(path==null) throw new NullPointerException();
        this.path = path;

        File baseDirectory = new File(path);
        if(!baseDirectory.isDirectory() ){
            throw new IllegalArgumentException("Invalid path: is not directory. " + path);
        }

        log.info("LocalMultiLayerNetworkSaver saving networks to local directory: {}",path);
    }

    @Override
    public void saveModel(OptimizationResult<MultiLayerConfiguration, MultiLayerNetwork> result) throws IOException {
        String dir = new File(path,result.getIndex() + "/").getAbsolutePath();

        File f = new File(dir);
        f.mkdir();

        String paramsPath = FilenameUtils.concat(dir,"params.bin");
        String jsonPath = FilenameUtils.concat(dir,"config.json");
        String scorePath = FilenameUtils.concat(dir,"score.txt");

        INDArray params = result.getResult().params();
        String jsonConfig = result.getConfig().toJson();

        FileUtils.writeStringToFile(new File(scorePath), String.valueOf(result.getScore()));
        FileUtils.writeStringToFile(new File(jsonPath), jsonConfig);
        try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(paramsPath)))){
            Nd4j.write(params, dos);
        }


        log.debug("Deeplearning4j model result (id={}, score={}) saved to directory: {}",result.getIndex(), result.getScore(), dir);
    }
}
