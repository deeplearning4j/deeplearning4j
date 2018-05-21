package org.deeplearning4j.ui.module.train;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.params.VariationalAutoencoderParamInitializer;

import java.util.*;

/**
 *
 * Utility methods for {@link TrainModule}
 *
 * @author Alex Black
 */
public class TrainModuleUtils {


    @AllArgsConstructor
    @Data
    public static class GraphInfo {

        private List<String> vertexNames;
        private List<String> vertexTypes;
        private List<List<Integer>> vertexInputs;
        private List<Map<String, String>> vertexInfo;

        @JsonIgnore
        private List<String> originalVertexName;
    }

    public static GraphInfo buildGraphInfo(MultiLayerConfiguration config) {
        List<String> vertexNames = new ArrayList<>();
        List<String> originalVertexName = new ArrayList<>();
        List<String> layerTypes = new ArrayList<>();
        List<List<Integer>> layerInputs = new ArrayList<>();
        List<Map<String, String>> layerInfo = new ArrayList<>();
        vertexNames.add("Input");
        originalVertexName.add(null);
        layerTypes.add("Input");
        layerInputs.add(Collections.emptyList());
        layerInfo.add(Collections.emptyMap());


        List<NeuralNetConfiguration> list = config.getConfs();
        int layerIdx = 1;
        for (NeuralNetConfiguration c : list) {
            Layer layer = c.getLayer();
            String layerName = layer.getLayerName();
            if (layerName == null)
                layerName = "layer" + layerIdx;
            vertexNames.add(layerName);
            originalVertexName.add(String.valueOf(layerIdx - 1));

            String layerType = c.getLayer().getClass().getSimpleName().replaceAll("Layer$", "");
            layerTypes.add(layerType);

            layerInputs.add(Collections.singletonList(layerIdx - 1));
            layerIdx++;

            //Extract layer info
            Map<String, String> map = getLayerInfo(c, layer);
            layerInfo.add(map);
        }

        return new GraphInfo(vertexNames, layerTypes, layerInputs, layerInfo, originalVertexName);
    }

    public static GraphInfo buildGraphInfo(ComputationGraphConfiguration config) {
        List<String> layerNames = new ArrayList<>();
        List<String> layerTypes = new ArrayList<>();
        List<List<Integer>> layerInputs = new ArrayList<>();
        List<Map<String, String>> layerInfo = new ArrayList<>();


        Map<String, GraphVertex> vertices = config.getVertices();
        Map<String, List<String>> vertexInputs = config.getVertexInputs();
        List<String> networkInputs = config.getNetworkInputs();

        List<String> originalVertexName = new ArrayList<>();

        Map<String, Integer> vertexToIndexMap = new HashMap<>();
        int vertexCount = 0;
        for (String s : networkInputs) {
            vertexToIndexMap.put(s, vertexCount++);
            layerNames.add(s);
            originalVertexName.add(s);
            layerTypes.add(s);
            layerInputs.add(Collections.emptyList());
            layerInfo.add(Collections.emptyMap());
        }

        for (String s : vertices.keySet()) {
            vertexToIndexMap.put(s, vertexCount++);
        }

        int layerCount = 0;
        for (Map.Entry<String, GraphVertex> entry : vertices.entrySet()) {
            GraphVertex gv = entry.getValue();
            layerNames.add(entry.getKey());

            List<String> inputsThisVertex = vertexInputs.get(entry.getKey());
            List<Integer> inputIndexes = new ArrayList<>();
            for (String s : inputsThisVertex) {
                inputIndexes.add(vertexToIndexMap.get(s));
            }

            layerInputs.add(inputIndexes);

            if (gv instanceof LayerVertex) {
                NeuralNetConfiguration c = ((LayerVertex) gv).getLayerConf();
                Layer layer = c.getLayer();

                String layerType = layer.getClass().getSimpleName().replaceAll("Layer$", "");
                layerTypes.add(layerType);

                //Extract layer info
                Map<String, String> map = getLayerInfo(c, layer);
                layerInfo.add(map);
            } else {
                String layerType = gv.getClass().getSimpleName();
                layerTypes.add(layerType);
                Map<String, String> thisVertexInfo = Collections.emptyMap(); //TODO
                layerInfo.add(thisVertexInfo);
            }
            originalVertexName.add(entry.getKey());
        }

        return new GraphInfo(layerNames, layerTypes, layerInputs, layerInfo, originalVertexName);
    }

    public static GraphInfo buildGraphInfo(NeuralNetConfiguration config) {

        List<String> vertexNames = new ArrayList<>();
        List<String> originalVertexName = new ArrayList<>();
        List<String> layerTypes = new ArrayList<>();
        List<List<Integer>> layerInputs = new ArrayList<>();
        List<Map<String, String>> layerInfo = new ArrayList<>();
        vertexNames.add("Input");
        originalVertexName.add(null);
        layerTypes.add("Input");
        layerInputs.add(Collections.emptyList());
        layerInfo.add(Collections.emptyMap());

        if (config.getLayer() instanceof VariationalAutoencoder) {
            //Special case like this is a bit ugly - but it works
            VariationalAutoencoder va = (VariationalAutoencoder) config.getLayer();
            int[] encLayerSizes = va.getEncoderLayerSizes();
            int[] decLayerSizes = va.getDecoderLayerSizes();

            int layerIndex = 1;
            for (int i = 0; i < encLayerSizes.length; i++) {
                String name = "encoder_" + i;
                vertexNames.add(name);
                originalVertexName.add("e" + i);
                String layerType = "VAE-Encoder";
                layerTypes.add(layerType);
                layerInputs.add(Collections.singletonList(layerIndex - 1));
                layerIndex++;

                Map<String, String> encoderInfo = new LinkedHashMap<>();
                long inputSize = (i == 0 ? va.getNIn() : encLayerSizes[i - 1]);
                long outputSize = encLayerSizes[i];
                encoderInfo.put("Input Size", String.valueOf(inputSize));
                encoderInfo.put("Layer Size", String.valueOf(outputSize));
                encoderInfo.put("Num Parameters", String.valueOf((inputSize + 1) * outputSize));
                encoderInfo.put("Activation Function", va.getActivationFn().toString());
                layerInfo.add(encoderInfo);
            }

            vertexNames.add("z");
            originalVertexName.add(VariationalAutoencoderParamInitializer.PZX_PREFIX);
            layerTypes.add("VAE-LatentVariable");
            layerInputs.add(Collections.singletonList(layerIndex - 1));
            layerIndex++;
            Map<String, String> latentInfo = new LinkedHashMap<>();
            long inputSize = encLayerSizes[encLayerSizes.length - 1];
            long outputSize = va.getNOut();
            latentInfo.put("Input Size", String.valueOf(inputSize));
            latentInfo.put("Layer Size", String.valueOf(outputSize));
            latentInfo.put("Num Parameters", String.valueOf((inputSize + 1) * outputSize * 2));
            latentInfo.put("Activation Function", va.getPzxActivationFn().toString());
            layerInfo.add(latentInfo);


            for (int i = 0; i < decLayerSizes.length; i++) {
                String name = "decoder_" + i;
                vertexNames.add(name);
                originalVertexName.add("d" + i);
                String layerType = "VAE-Decoder";
                layerTypes.add(layerType);
                layerInputs.add(Collections.singletonList(layerIndex - 1));
                layerIndex++;

                Map<String, String> decoderInfo = new LinkedHashMap<>();
                inputSize = (i == 0 ? va.getNOut() : decLayerSizes[i - 1]);
                outputSize = decLayerSizes[i];
                decoderInfo.put("Input Size", String.valueOf(inputSize));
                decoderInfo.put("Layer Size", String.valueOf(outputSize));
                decoderInfo.put("Num Parameters", String.valueOf((inputSize + 1) * outputSize));
                decoderInfo.put("Activation Function", va.getActivationFn().toString());
                layerInfo.add(decoderInfo);
            }

            vertexNames.add("x");
            originalVertexName.add(VariationalAutoencoderParamInitializer.PXZ_PREFIX);
            layerTypes.add("VAE-Reconstruction");
            layerInputs.add(Collections.singletonList(layerIndex - 1));
            layerIndex++;
            Map<String, String> reconstructionInfo = new LinkedHashMap<>();
            inputSize = decLayerSizes[decLayerSizes.length - 1];
            outputSize = va.getNIn();
            reconstructionInfo.put("Input Size", String.valueOf(inputSize));
            reconstructionInfo.put("Layer Size", String.valueOf(outputSize));
            reconstructionInfo.put("Num Parameters", String
                    .valueOf((inputSize + 1) * va.getOutputDistribution().distributionInputSize((int) va.getNIn())));
            reconstructionInfo.put("Distribution", va.getOutputDistribution().toString());
            layerInfo.add(reconstructionInfo);


        } else {
            //VAE or similar...
            Layer layer = config.getLayer();
            String layerName = layer.getLayerName();
            if (layerName == null)
                layerName = "layer0";
            vertexNames.add(layerName);
            originalVertexName.add(String.valueOf("0"));

            String layerType = config.getLayer().getClass().getSimpleName().replaceAll("Layer$", "");
            layerTypes.add(layerType);

            layerInputs.add(Collections.singletonList(0));

            //Extract layer info
            Map<String, String> map = getLayerInfo(config, layer);
            layerInfo.add(map);
        }


        return new GraphInfo(vertexNames, layerTypes, layerInputs, layerInfo, originalVertexName);
    }


    private static Map<String, String> getLayerInfo(NeuralNetConfiguration c, Layer layer) {

        Map<String, String> map = new LinkedHashMap<>();

        if (layer instanceof FeedForwardLayer) {
            FeedForwardLayer layer1 = (FeedForwardLayer) layer;
            map.put("Input size", String.valueOf(layer1.getNIn()));
            map.put("Output size", String.valueOf(layer1.getNOut()));
            map.put("Num Parameters", String.valueOf(layer1.initializer().numParams(c)));
            map.put("Activation Function", layer1.getActivationFn().toString());
        }

        if (layer instanceof ConvolutionLayer) {
            org.deeplearning4j.nn.conf.layers.ConvolutionLayer layer1 =
                    (org.deeplearning4j.nn.conf.layers.ConvolutionLayer) layer;
            map.put("Kernel size", Arrays.toString(layer1.getKernelSize()));
            map.put("Stride", Arrays.toString(layer1.getStride()));
            map.put("Padding", Arrays.toString(layer1.getPadding()));
        } else if (layer instanceof SubsamplingLayer) {
            SubsamplingLayer layer1 = (SubsamplingLayer) layer;
            map.put("Kernel size", Arrays.toString(layer1.getKernelSize()));
            map.put("Stride", Arrays.toString(layer1.getStride()));
            map.put("Padding", Arrays.toString(layer1.getPadding()));
            map.put("Pooling Type", layer1.getPoolingType().toString());
        } else if (layer instanceof BaseOutputLayer) {
            BaseOutputLayer ol = (BaseOutputLayer) layer;
            if(ol.getLossFn() != null)
                map.put("Loss Function", ol.getLossFn().toString());
        }

        return map;
    }
}
