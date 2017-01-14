package org.deeplearning4j.nn.modelimport.keras.trainedmodels;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * @author susaneraly
 */
public class TrainedModelHelper {

    protected static final Logger logger = LoggerFactory.getLogger(TrainedModelHelper.class);

    private final String h5URL;
    private final String jsonURL;

    private static final File HOME_DIR = new File(System.getProperty("user.home"));
    private static final String BASE_DIR = ".dl4j/trainedmodels/";
    private final File MODEL_DIR;
    private final String h5FileName;
    private final String jsonFileName;

    private File h5File;
    private File jsonFile;
    private boolean userProvidedPath = false;

    private final DataSetPreProcessor PRE_PROCESSOR;

    public TrainedModelHelper(TrainedModels model) {
        this.h5URL = model.getH5URL();
        this.jsonURL = model.getJSONURL();
        this.h5FileName = model.getH5FileName();
        this.jsonFileName = model.getJSONFileName();
        this.MODEL_DIR = new File(HOME_DIR, BASE_DIR + model.getModelDir());
        this.h5File = new File(MODEL_DIR,h5FileName);
        this.jsonFile = new File(MODEL_DIR,jsonFileName);
        this.PRE_PROCESSOR = model.getPreProcessor();
    }

    public void setPathToH5(String pathtoH5) {
        this.h5File = new File(pathtoH5);
        this.userProvidedPath = true;
        logger.info("Helper will use path given to H5 file");
    }

    public ComputationGraph loadModel() throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        download();
        return KerasModelImport.importKerasModelAndWeights(jsonFile.getAbsolutePath(), h5File.getAbsolutePath());
    }

    private void download() throws IOException {
        if (!h5File.exists() && !userProvidedPath) {
            if (!(MODEL_DIR.isDirectory() || MODEL_DIR.mkdirs())) {
                throw new IOException("Could not mkdir " + MODEL_DIR);
            }
            logger.info("H5 weights not found in default location. Copying from URL "+h5URL+"\n\tto location "+h5File.getAbsolutePath());
            FileUtils.copyURLToFile(new URL(h5URL), h5File);
        }
        if (!jsonFile.exists()) {
            if (!(MODEL_DIR.isDirectory() || MODEL_DIR.mkdirs())) {
                throw new IOException("Could not mkdir " + MODEL_DIR);
            }
            logger.info("JSON config not found in default location. Copying from URL "+jsonURL+"\n\tto location "+ jsonFile.getAbsolutePath());
            FileUtils.copyURLToFile(new URL(jsonURL), jsonFile);
        }
    }

    public DataSetPreProcessor getPreProcessor() {
        return this.PRE_PROCESSOR;
    }
}
