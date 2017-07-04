package org.deeplearning4j.nn.conf.serde;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class ComputationGraphConfigurationDeserializer
                extends BaseNetConfigDeserializer<ComputationGraphConfiguration> {

    public ComputationGraphConfigurationDeserializer(JsonDeserializer<?> defaultDeserializer) {
        super(defaultDeserializer, ComputationGraphConfiguration.class);
    }

    @Override
    public ComputationGraphConfiguration deserialize(JsonParser jp, DeserializationContext ctxt)
                    throws IOException, JsonProcessingException {
        ComputationGraphConfiguration conf = (ComputationGraphConfiguration) defaultDeserializer.deserialize(jp, ctxt);

        //Updater configuration changed after 0.8.0 release
        //Previously: enumerations and fields. Now: classes
        //Here, we manually create the appropriate Updater instances, if the IUpdater field is empty

        List<Layer> layerList = new ArrayList<>();
        Map<String, GraphVertex> vertices = conf.getVertices();
        for (Map.Entry<String, GraphVertex> entry : vertices.entrySet()) {
            if (entry.getValue() instanceof LayerVertex) {
                LayerVertex lv = (LayerVertex) entry.getValue();
                layerList.add(lv.getLayerConf().getLayer());
            }
        }

        Layer[] layers = layerList.toArray(new Layer[layerList.size()]);
        handleUpdaterBackwardCompatibility(layers);

        return conf;
    }
}
