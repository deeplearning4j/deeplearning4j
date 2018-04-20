package org.deeplearning4j.nn.conf.serde;

import org.deeplearning4j.nn.conf.json.JsonMappers;
import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
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
//                @JsonSubTypes.Type(value = SubsamplingLayer.class, name = "subsampling"),
//                @JsonSubTypes.Type(value = Subsampling1DLayer.class, name = "subsampling1d"),
//                @JsonSubTypes.Type(value = BatchNormalization.class, name = "batchNormalization"),
//                @JsonSubTypes.Type(value = LocalResponseNormalization.class, name = "localResponseNormalization"),
//                @JsonSubTypes.Type(value = EmbeddingLayer.class, name = "embedding"),
//                @JsonSubTypes.Type(value = ActivationLayer.class, name = "activation"),
//                @JsonSubTypes.Type(value = VariationalAutoencoder.class, name = "VariationalAutoencoder"),
//                @JsonSubTypes.Type(value = DropoutLayer.class, name = "dropout"),
//                @JsonSubTypes.Type(value = GlobalPoolingLayer.class, name = "GlobalPooling"),
//                @JsonSubTypes.Type(value = ZeroPaddingLayer.class, name = "zeroPadding"),
//                @JsonSubTypes.Type(value = ZeroPadding1DLayer.class, name = "zeroPadding1d"),
//                @JsonSubTypes.Type(value = FrozenLayer.class, name = "FrozenLayer"),
//                @JsonSubTypes.Type(value = Upsampling2D.class, name = "Upsampling2D"),
//                @JsonSubTypes.Type(value = Yolo2OutputLayer.class, name = "Yolo2OutputLayer"),
//                @JsonSubTypes.Type(value = RnnLossLayer.class, name = "RnnLossLayer"),
//                @JsonSubTypes.Type(value = CnnLossLayer.class, name = "CnnLossLayer"),
//                @JsonSubTypes.Type(value = Bidirectional.class, name = "Bidirectional"),
//                @JsonSubTypes.Type(value = SimpleRnn.class, name = "SimpleRnn"),
//                @JsonSubTypes.Type(value = ElementWiseMultiplicationLayer.class, name = "ElementWiseMult"),
//                @JsonSubTypes.Type(value = MaskLayer.class, name = "MaskLayer"),
//                @JsonSubTypes.Type(value = MaskZeroLayer.class, name = "MaskZeroLayer"),
//                @JsonSubTypes.Type(value = Cropping1D.class, name = "Cropping1D"),
//                @JsonSubTypes.Type(value = Cropping2D.class, name = "Cropping2D")}
    }



    @Override
    public Layer deserialize(JsonParser jp, DeserializationContext deserializationContext) throws IOException, JsonProcessingException {
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
        Layer l;
        try {
            l = m.readValue(nodeAsString, lClass);
        } catch (Exception e){
            throw e;
        }
        return l;
    }



}
