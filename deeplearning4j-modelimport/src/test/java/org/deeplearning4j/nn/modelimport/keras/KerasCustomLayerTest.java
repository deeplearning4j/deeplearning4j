package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.layers.*;
import org.deeplearning4j.nn.modelimport.keras.layers.custom.KerasLRN;
import org.deeplearning4j.nn.modelimport.keras.layers.custom.KerasPoolHelper;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_CLASS_NAME_ACTIVATION;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_CLASS_NAME_CONVOLUTION_2D;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_CLASS_NAME_DENSE;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_CLASS_NAME_DROPOUT;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_CLASS_NAME_LSTM;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_CLASS_NAME_MAX_POOLING_2D;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_ACTIVATION;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_BORDER_MODE;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_CLASS_NAME;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_CONFIG;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_DROPOUT;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_DROPOUT_W;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_INIT;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_NAME;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_NB_COL;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_NB_FILTER;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_NB_ROW;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_OUTPUT_DIM;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_POOL_SIZE;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_STRIDES;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_SUBSAMPLE;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.LAYER_FIELD_W_REGULARIZER;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.REGULARIZATION_TYPE_L1;
import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.REGULARIZATION_TYPE_L2;
import static org.deeplearning4j.nn.modelimport.keras.layers.KerasBatchNormalization.*;
import static org.deeplearning4j.nn.modelimport.keras.layers.KerasLstm.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test import of Keras custom layers. Must be run manually, since user must download weights and config from
 * http://blob.deeplearning4j.org/models/googlenet_keras_weights.h5
 * http://blob.deeplearning4j.org/models/googlenet_config.json
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class KerasCustomLayerTest {

    // run manually
    @Test
    public void testCustomLayerImport() throws Exception {
        // file paths
        String kerasWeightsAndConfigUrl = "http://blob.deeplearning4j.org/models/googlenet_keras_weightsandconfig.h5";
        File cachedKerasFile = new File(System.getProperty("user.tmp"),"googlenet_keras_weightsandconfig.h5");
        String outputPath = "/home/justin/Downloads/googlenet_dl4j_inference.zip";

        KerasLayer.registerCustomLayer("PoolHelper", KerasPoolHelper.class);
        KerasLayer.registerCustomLayer("LRN", KerasLRN.class);

        // download file
        if (!cachedKerasFile.exists()) {
            log.info("Downloading model to " + cachedKerasFile.toString());
            FileUtils.copyURLToFile(new URL(kerasWeightsAndConfigUrl), cachedKerasFile);
            cachedKerasFile.deleteOnExit();
        }

        org.deeplearning4j.nn.api.Model importedModel = KerasModelImport.importKerasModelAndWeights(cachedKerasFile.getAbsolutePath());
        ModelSerializer.writeModel(importedModel, outputPath, false);

        ComputationGraph serializedModel = ModelSerializer.restoreComputationGraph(outputPath);
        log.info(serializedModel.summary());
    }
}
