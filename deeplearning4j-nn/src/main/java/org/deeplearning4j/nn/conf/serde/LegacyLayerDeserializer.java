package org.deeplearning4j.nn.conf.serde;

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
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.*;

public class LegacyLayerDeserializer extends JsonDeserializer<Layer> {

    private static final Map<String,String> LEGACY_NAMES = new HashMap<>();

    static {
        LEGACY_NAMES.put("autoEncoder", AutoEncoder.class.getName());
        LEGACY_NAMES.put("convolution", ConvolutionLayer.class.getName());
        LEGACY_NAMES.put("convolution1d", Convolution1DLayer.class.getName());
        LEGACY_NAMES.put("gravesLSTM", GravesLSTM.class.getName());
        LEGACY_NAMES.put("LSTM", LSTM.class.getName());
        LEGACY_NAMES.put("gravesBidirectionalLSTM", GravesBidirectionalLSTM.class.getName());
        LEGACY_NAMES.put("output", OutputLayer.class.getName());
        LEGACY_NAMES.put("CenterLossOutputLayer", CenterLossOutputLayer.class.getName());
        LEGACY_NAMES.put("rnnoutput", RnnOutputLayer.class.getName());
        LEGACY_NAMES.put("loss", LossLayer.class.getName());
        LEGACY_NAMES.put("dense", DenseLayer.class.getName());
        LEGACY_NAMES.put("subsampling", SubsamplingLayer.class.getName());
        LEGACY_NAMES.put("subsampling1d", Subsampling1DLayer.class.getName());
        LEGACY_NAMES.put("batchNormalization", BatchNormalization.class.getName());
        LEGACY_NAMES.put("localResponseNormalization", LocalResponseNormalization.class.getName());
        LEGACY_NAMES.put("embedding", EmbeddingLayer.class.getName());
        LEGACY_NAMES.put("activation", VariationalAutoencoder.class.getName());
        LEGACY_NAMES.put("dropout", DropoutLayer.class.getName());
        LEGACY_NAMES.put("GlobalPooling", GlobalPoolingLayer.class.getName());
        LEGACY_NAMES.put("zeroPadding", ZeroPaddingLayer.class.getName());
        LEGACY_NAMES.put("zeroPadding1d", ZeroPadding1DLayer.class.getName());
        LEGACY_NAMES.put("FrozenLayer", FrozenLayer.class.getName());
        LEGACY_NAMES.put("Upsampling2D", Upsampling2D.class.getName());
        LEGACY_NAMES.put("Yolo2OutputLayer", Yolo2OutputLayer.class.getName());
        LEGACY_NAMES.put("RnnLossLayer", RnnLossLayer.class.getName());
        LEGACY_NAMES.put("CnnLossLayer", CnnLossLayer.class.getName());
        LEGACY_NAMES.put("Bidirectional", Bidirectional.class.getName());
        LEGACY_NAMES.put("SimpleRnn", SimpleRnn.class.getName());
        LEGACY_NAMES.put("ElementWiseMult", ElementWiseMultiplicationLayer.class.getName());
        LEGACY_NAMES.put("MaskLayer", MaskLayer.class.getName());
        LEGACY_NAMES.put("MaskZeroLayer", MaskZeroLayer.class.getName());
        LEGACY_NAMES.put("Cropping1D", Cropping1D.class.getName());
        LEGACY_NAMES.put("Cropping2D", Cropping2D.class.getName());

        //The following didn't previously have subtype annotations - hence will be using default name (class simple name)
        LEGACY_NAMES.put("LastTimeStep", LastTimeStep.class.getName());


        //TODO: Keras layers
    }

    @Override
    public Layer deserialize(JsonParser jp, DeserializationContext deserializationContext) throws IOException {
        //Manually parse old format
        JsonNode node = jp.getCodec().readTree(jp);

        Iterator<Map.Entry<String,JsonNode>> nodes = node.fields();
        //For legacy format, ex

        List<Map.Entry<String,JsonNode>> list = new ArrayList<>();
        while(nodes.hasNext()){
            list.add(nodes.next());
        }

        if(list.size() != 1){
            throw new IllegalStateException("Expected size 1: " + list.size());
        }

        String name = list.get(0).getKey();
        JsonNode value = list.get(0).getValue();

        String layerClass = LEGACY_NAMES.get(name);
        if(layerClass == null){
            throw new IllegalStateException("Cannot deserialize: " + name);
        }

        Class<? extends Layer> lClass;
        try {
            lClass = (Class<? extends Layer>) Class.forName(layerClass);
        } catch (Exception e){
            throw new RuntimeException(e);
        }

        ObjectMapper m = JsonMappers.getMapperLegacyJson();

        String nodeAsString = value.toString();
        Layer l = m.readValue(nodeAsString, lClass);
        return l;
    }



}
