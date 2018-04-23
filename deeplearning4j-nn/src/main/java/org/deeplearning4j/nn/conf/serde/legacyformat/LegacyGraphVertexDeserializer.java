package org.deeplearning4j.nn.conf.serde.legacyformat;

import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.graph.rnn.ReverseTimeSeriesVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.layers.util.MaskLayer;
import org.deeplearning4j.nn.conf.layers.util.MaskZeroLayer;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.*;

public class LegacyGraphVertexDeserializer extends BaseLegacyDeserializer<GraphVertex> {

    private static final Map<String,String> LEGACY_NAMES = new HashMap<>();

    static {


        List<Class<? extends GraphVertex>> cList = Arrays.asList(
                //All of these vertices had the legacy format name the same as the simple class name
                MergeVertex.class,
                SubsetVertex.class,
                LayerVertex.class,
                LastTimeStepVertex.class,
                ReverseTimeSeriesVertex.class,
                DuplicateToTimeSeriesVertex.class,
                PreprocessorVertex.class,
                StackVertex.class,
                UnstackVertex.class,
                L2Vertex.class,
                ScaleVertex.class,
                L2NormalizeVertex.class,
                //These did not previously have a subtype annotation - they use default (which is simple class name)
                ElementWiseVertex.class,
                PoolHelperVertex.class,
                ReshapeVertex.class,
                ShiftVertex.class);

        for(Class<?> c : cList){
            LEGACY_NAMES.put(c.getSimpleName(), c.getName());
        }


        //TODO: Keras vertices - if any?
    }


    @Override
    public Map<String, String> getLegacyNamesMap() {
        return LEGACY_NAMES;
    }
}
