package org.deeplearning4j.util;

import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.ModelConfiguration;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.UUID;

import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 12/29/16.
 */
public class ModelGuesserTest {

    @Test
    public void testModelGuess() throws Exception {
        ClassPathResource resource = new ClassPathResource("/keras/simple/cnn_tf_config.json");
        File f = getTempFile(resource);
        String configFilename = f.getAbsolutePath();
        Object conf = ModelGuesser.loadConfigGuess(configFilename);
        assertTrue(conf instanceof MultiLayerConfiguration);

        ClassPathResource sequenceResource = new ClassPathResource("keras/simple/mlp_fapi_multiloss_config.json");
        File f2 = getTempFile(sequenceResource);
        Object sequenceConf = ModelGuesser.loadConfigGuess(f2.getAbsolutePath());
        assertTrue(sequenceConf instanceof ComputationGraphConfiguration);
    }


    private File getTempFile(ClassPathResource classPathResource) throws Exception {
        InputStream is = classPathResource.getInputStream();
        File f = new File(UUID.randomUUID().toString());
        f.deleteOnExit();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f));
        IOUtils.copy(is,bos);
        bos.flush();
        bos.close();
        return f;
    }

}
