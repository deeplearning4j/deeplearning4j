package org.deeplearning4j.nn.modelimport.keras.configurations;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.core.KerasDense;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class KerasInitilizationTest {

    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    @Test
    public void testInitializers() throws Exception {

        Integer keras1 = 1;
        Integer keras2 = 2;

        String[] keras1Inits = initializers(conf1);
        String[] keras2Inits = initializers(conf2);
        WeightInit[] dl4jInits = dl4jInitializers();

        for (int i=0; i< dl4jInits.length; i++) {
            initilizationDenseLayer(conf1, keras1, keras1Inits[i], dl4jInits[i]);
            initilizationDenseLayer(conf2, keras2,  keras2Inits[i], dl4jInits[i]);
        }
    }

    private String[] initializers(KerasLayerConfiguration conf) {
        return new String[] {
                conf.getINIT_GLOROT_NORMAL(),
                conf.getINIT_GLOROT_UNIFORM(),
                conf.getINIT_LECUN_NORMAL(),
                conf.getINIT_RANDOM_UNIFORM(),
                conf.getINIT_HE_NORMAL(),
                conf.getINIT_HE_UNIFORM(),
                conf.getINIT_ONES(),
                conf.getINIT_ZERO(),
                conf.getINIT_IDENTITY(),
                conf.getINIT_VARIANCE_SCALING()
                // TODO: add these initializations
                // conf.getINIT_CONSTANT(),
                // conf.getINIT_NORMAL(),
                // conf.getINIT_ORTHOGONAL(),
                // conf.getINIT_LECUN_UNIFORM()
        };
    }

    private WeightInit[] dl4jInitializers() {
        return new WeightInit[] {
                WeightInit.XAVIER,
                WeightInit.XAVIER_UNIFORM,
                WeightInit.LECUN_NORMAL,
                WeightInit.UNIFORM,
                WeightInit.RELU,
                WeightInit.RELU_UNIFORM,
                WeightInit.ONES,
                WeightInit.ZERO,
                WeightInit.IDENTITY,
                WeightInit.XAVIER_UNIFORM // TODO: Variance scaling is incorrectly mapped
        };
    }
    
    private void initilizationDenseLayer(KerasLayerConfiguration conf, Integer kerasVersion,
                                 String initializer, WeightInit dl4jInitializer)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_DENSE());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_ACTIVATION(), "linear");
        config.put(conf.getLAYER_FIELD_NAME(), "init_test");
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_INIT(), initializer);
        } else {
            Map<String, Object> init = new HashMap<>();
            init.put("class_name", initializer);
            config.put(conf.getLAYER_FIELD_INIT(), init);
        }
        config.put(conf.getLAYER_FIELD_OUTPUT_DIM(), 1337);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        DenseLayer layer = new KerasDense(layerConfig, false).getDenseLayer();
        assertEquals(dl4jInitializer, layer.getWeightInit());
    }
}
