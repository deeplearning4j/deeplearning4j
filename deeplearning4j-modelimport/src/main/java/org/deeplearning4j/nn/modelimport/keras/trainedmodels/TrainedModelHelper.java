package org.deeplearning4j.nn.modelimport.keras.trainedmodels;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * @author susaneraly
 * @deprecated Please use the new module deeplearning4j-zoo and instantiate pretrained models from the zoo directly.
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
    private boolean userProvidedH5 = false;
    private boolean userProvidedJSON = false;

    private String[] decodeMap;

    public TrainedModelHelper(TrainedModels model) {
        this.MODEL_DIR = new File(HOME_DIR, BASE_DIR + model.getModelDir());

        this.h5URL = model.getH5URL();
        this.h5FileName = model.getH5FileName();
        this.h5File = new File(MODEL_DIR, h5FileName);

        this.jsonURL = model.getJSONURL();
        this.jsonFileName = model.getJSONFileName();
        this.jsonFile = new File(MODEL_DIR, jsonFileName);
    }

    public void setPathToH5(String pathtoH5) {
        this.h5File = new File(pathtoH5);
        this.userProvidedH5 = true;
        logger.info("Helper will use path given to H5 file");
    }

    public void setPathToJSON(String pathToJSON) {
        this.jsonFile = new File(pathToJSON);
        this.userProvidedJSON = true;
        logger.info("Helper will use path given to JSON file");
    }

    /*
        FIXME
        Once I upload the file this should be a static class - leaving as is for now
        And the set paths methods above must go
     */
    public ComputationGraph loadModel()
                    throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        download();
        return KerasModelImport.importKerasModelAndWeights(jsonFile.getAbsolutePath(), h5File.getAbsolutePath(), false);
    }

    private void download() throws IOException {
        if (!h5File.exists() && !userProvidedH5) {
            if (!(MODEL_DIR.isDirectory() || MODEL_DIR.mkdirs())) {
                throw new IOException("Could not mkdir " + MODEL_DIR);
            }
            logger.info("H5 weights not found in default location. Copying from URL " + h5URL + "\n\tto location "
                            + h5File.getAbsolutePath());
            FileUtils.copyURLToFile(new URL(h5URL), h5File);
        }
        if (!jsonFile.exists() && !userProvidedJSON) {
            if (!(MODEL_DIR.isDirectory() || MODEL_DIR.mkdirs())) {
                throw new IOException("Could not mkdir " + MODEL_DIR);
            }
            logger.info("JSON config not found in default location. Copying from URL " + jsonURL + "\n\tto location "
                            + jsonFile.getAbsolutePath());
            FileUtils.copyURLToFile(new URL(jsonURL), jsonFile);
        }
    }

}
