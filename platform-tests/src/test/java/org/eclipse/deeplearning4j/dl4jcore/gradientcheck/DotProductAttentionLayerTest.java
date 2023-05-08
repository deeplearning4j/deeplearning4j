package org.eclipse.deeplearning4j.dl4jcore.gradientcheck;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DotProductAttentionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

@DisplayName("Dot Product Attention Test")
@Tag(TagNames.NDARRAY_ETL)
@Tag(TagNames.TRAINING)
@Tag(TagNames.DL4J_OLD_API)
@NativeTag
@Tag(TagNames.LARGE_RESOURCES)
@Tag(TagNames.LONG_TEST)
public class DotProductAttentionLayerTest extends BaseDL4JTest  {


    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }



    @Test
    public void testDotProductAttention() {
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .list(new DotProductAttentionLayer.Builder()
                        .nIn(10)
                        .nOut(10)
                        .dropOut(0.5)
                        .scale(0.5)
                        .build())
                .build();

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(multiLayerConfiguration);
    }

}
