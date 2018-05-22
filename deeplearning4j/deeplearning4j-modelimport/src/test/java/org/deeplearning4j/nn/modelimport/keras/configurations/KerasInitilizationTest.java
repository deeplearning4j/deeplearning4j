package org.deeplearning4j.nn.modelimport.keras.configurations;

import org.deeplearning4j.nn.conf.distribution.*;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.core.KerasDense;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.primitives.Pair;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class KerasInitilizationTest {

    private double minValue = -0.2;
    private double maxValue = 0.2;
    private double mean = 0.0;
    private double stdDev = 0.2;
    private double value = 42.0;
    private double gain = 0.2;

    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    @Test
    public void testInitializers() throws Exception {

        Integer keras1 = 1;
        Integer keras2 = 2;

        String[] keras1Inits = initializers(conf1);
        String[] keras2Inits = initializers(conf2);
        WeightInit[] dl4jInits = dl4jInitializers();
        Distribution[] dl4jDistributions = dl4jDistributions();

        for (int i = 0; i < dl4jInits.length - 1; i++) {
            initilizationDenseLayer(conf1, keras1, keras1Inits[i], dl4jInits[i], dl4jDistributions[i]);
            initilizationDenseLayer(conf2, keras2, keras2Inits[i], dl4jInits[i], dl4jDistributions[i]);

            initilizationDenseLayer(conf2, keras2, keras2Inits[dl4jInits.length - 1],
                    dl4jInits[dl4jInits.length - 1], dl4jDistributions[dl4jInits.length - 1]);
        }
    }

    private String[] initializers(KerasLayerConfiguration conf) {
        return new String[]{
                conf.getINIT_GLOROT_NORMAL(),
                conf.getINIT_GLOROT_UNIFORM(),
                conf.getINIT_LECUN_NORMAL(),
                conf.getINIT_LECUN_UNIFORM(),
                conf.getINIT_RANDOM_UNIFORM(),
                conf.getINIT_HE_NORMAL(),
                conf.getINIT_HE_UNIFORM(),
                conf.getINIT_ONES(),
                conf.getINIT_ZERO(),
                conf.getINIT_IDENTITY(),
                conf.getINIT_NORMAL(),
                conf.getINIT_ORTHOGONAL(),
                conf.getINIT_CONSTANT(),
                conf.getINIT_VARIANCE_SCALING()

        };
    }

    private WeightInit[] dl4jInitializers() {
        return new WeightInit[]{
                WeightInit.XAVIER,
                WeightInit.XAVIER_UNIFORM,
                WeightInit.LECUN_NORMAL,
                WeightInit.LECUN_UNIFORM,
                WeightInit.DISTRIBUTION,
                WeightInit.RELU,
                WeightInit.RELU_UNIFORM,
                WeightInit.ONES,
                WeightInit.ZERO,
                WeightInit.IDENTITY,
                WeightInit.DISTRIBUTION,
                WeightInit.DISTRIBUTION,
                WeightInit.DISTRIBUTION,
                WeightInit.VAR_SCALING_NORMAL_FAN_IN};
    }

    private Distribution[] dl4jDistributions() {
        return new Distribution[]{
                null,
                null,
                null,
                null,
                new UniformDistribution(minValue, maxValue),
                null,
                null,
                null,
                null,
                null,
                new NormalDistribution(mean, stdDev),
                new OrthogonalDistribution(gain),
                new ConstantDistribution(value),
                null};
    }

    private void initilizationDenseLayer(KerasLayerConfiguration conf, Integer kerasVersion,
                                         String initializer, WeightInit dl4jInitializer, Distribution dl4jDistribution)
            throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_DENSE());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_ACTIVATION(), "linear");
        config.put(conf.getLAYER_FIELD_NAME(), "init_test");
        double scale = 0.2;
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_INIT(), initializer);
            config.put(conf.getLAYER_FIELD_INIT_MEAN(), mean);
            config.put(conf.getLAYER_FIELD_INIT_STDDEV(), stdDev);
            config.put(conf.getLAYER_FIELD_INIT_SCALE(), scale);
            config.put(conf.getLAYER_FIELD_INIT_MINVAL(), minValue);
            config.put(conf.getLAYER_FIELD_INIT_MAXVAL(), maxValue);
            config.put(conf.getLAYER_FIELD_INIT_VALUE(), value);
            config.put(conf.getLAYER_FIELD_INIT_GAIN(), gain);
        } else {
            Map<String, Object> init = new HashMap<>();
            init.put("class_name", initializer);
            Map<String, Object> innerInit = new HashMap<>();
            innerInit.put(conf.getLAYER_FIELD_INIT_MEAN(), mean);
            innerInit.put(conf.getLAYER_FIELD_INIT_STDDEV(), stdDev);
            innerInit.put(conf.getLAYER_FIELD_INIT_SCALE(), scale);
            innerInit.put(conf.getLAYER_FIELD_INIT_MINVAL(), minValue);
            innerInit.put(conf.getLAYER_FIELD_INIT_MAXVAL(), maxValue);
            innerInit.put(conf.getLAYER_FIELD_INIT_VALUE(), value);
            innerInit.put(conf.getLAYER_FIELD_INIT_GAIN(), gain);
            String mode = "fan_in";
            innerInit.put(conf.getLAYER_FIELD_INIT_MODE(), mode);
            String distribution = "normal";
            innerInit.put(conf.getLAYER_FIELD_INIT_DISTRIBUTION(), distribution);

            init.put(conf.getLAYER_FIELD_CONFIG(), innerInit);
            config.put(conf.getLAYER_FIELD_INIT(), init);
        }
        config.put(conf.getLAYER_FIELD_OUTPUT_DIM(), 1337);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        DenseLayer layer = new KerasDense(layerConfig, false).getDenseLayer();
        assertEquals(dl4jInitializer, layer.getWeightInit());
        assertEquals(dl4jDistribution, layer.getDist());

    }
}
