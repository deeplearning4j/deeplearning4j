package org.deeplearning4j.nn.conf.serde;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;

import java.io.IOException;

public class MultiLayerConfigurationDeserializer extends BaseNetConfigDeserializer<MultiLayerConfiguration> {

    public MultiLayerConfigurationDeserializer(JsonDeserializer<?> defaultDeserializer) {
        super(defaultDeserializer, MultiLayerConfiguration.class);
    }

    @Override
    public MultiLayerConfiguration deserialize(JsonParser jp, DeserializationContext ctxt)
                    throws IOException, JsonProcessingException {
        MultiLayerConfiguration conf = (MultiLayerConfiguration) defaultDeserializer.deserialize(jp, ctxt);

        //Updater configuration changed after 0.8.0 release
        //Previously: enumerations and fields. Now: classes
        //Here, we manually create the appropriate Updater instances, if the IUpdater field is empty

        Layer[] layers = new Layer[conf.getConfs().size()];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = conf.getConf(i).getLayer();
        }

        handleUpdaterBackwardCompatibility(layers);

        return conf;
    }
}
