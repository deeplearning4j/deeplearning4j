package org.arbiter.deeplearning4j.saver.local;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.saving.ResultReference;
import org.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

/**Basic MultiLayerNetwork saver. Saves config, parameters and score to: baseDir/0/, baseDir/1/, etc
 * where index is given by OptimizationResult.getIndex()
 */
public class LocalMultiLayerNetworkSaver<A> implements ResultSaver<MultiLayerConfiguration,MultiLayerNetwork,A> {
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
    public ResultReference<MultiLayerConfiguration,MultiLayerNetwork,A> saveModel(OptimizationResult<MultiLayerConfiguration, MultiLayerNetwork, A> result) throws IOException {
        String dir = new File(path,result.getIndex() + "/").getAbsolutePath();

        File f = new File(dir);
        f.mkdir();

        File paramsFile = new File(FilenameUtils.concat(dir,"params.bin"));
        File jsonFile = new File(FilenameUtils.concat(dir,"config.json"));
        File scoreFile = new File(FilenameUtils.concat(dir,"score.txt"));
        File additionalResultsFile = new File(FilenameUtils.concat(dir,"additionalResults.bin"));

        INDArray params = result.getResult().params();
        String jsonConfig = result.getCandidate().getValue().toJson();

        FileUtils.writeStringToFile(scoreFile, String.valueOf(result.getScore()));
        FileUtils.writeStringToFile(jsonFile, jsonConfig);
        try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(paramsFile.toPath()))){
            Nd4j.write(params, dos);
        }

        A additionalResults = result.getModelSpecificResults();
        if(additionalResults != null) {
            try(ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(additionalResultsFile))) {
                oos.writeObject(additionalResults);
            }
        }

        log.debug("Deeplearning4j model result (id={}, score={}) saved to directory: {}",result.getIndex(), result.getScore(), dir);

        return new LocalFileMultiLayerNetworkResultReference(jsonFile,paramsFile,scoreFile,additionalResultsFile,result.getCandidate());
    }
}
