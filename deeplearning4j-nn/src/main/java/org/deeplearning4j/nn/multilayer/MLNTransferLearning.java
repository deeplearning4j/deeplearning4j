package org.deeplearning4j.nn.multilayer;

import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Triple;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Other things to consider:
 * - There really should be a way to featurize and save to disk and then train from the featurized data. This will help users iterate quicker and
 * get a "nano net" that converges fast and then they can "fineTune" to their heart's content without wondering about having disruptive gradients
 * flowing backward to the unfrozen layers.
 * - And then adapting this for computation graphs (yikes)
 * - Also a summary of the model before and after to show how many new params were added/deleted and how many are learnable and how many are frozen etc..
 */
public class MLNTransferLearning {

    private MultiLayerConfiguration origConf;
    private INDArray origParams;

    public static class Builder {

        private INDArray origParams;
        private int tillIndex = -1;
        private List<INDArray> appendParams = new ArrayList<>(); //these could be new arrays, and views from origParams
        private MultiLayerConfiguration origConf;
        private MultiLayerNetwork origModel;
        private MultiLayerNetwork editedModel;
        private NeuralNetConfiguration.Builder globalConfig;

        private int frozenTill = -1;
        private int popFrom = -1;
        private boolean prepDone = false;
        private List<Integer> editedLayers = new ArrayList<>();
        private Map<Integer, Triple<Integer,WeightInit,WeightInit>> editedLayersMap = new HashMap<>();
        private List<NeuralNetConfiguration> editedConfs = new ArrayList<>();

        protected Map<Integer, InputPreProcessor> inputPreProcessors = new HashMap<>();
        protected boolean pretrain = false;
        protected boolean backprop = true;
        protected BackpropType backpropType = BackpropType.Standard;
        protected int tbpttFwdLength = 20;
        protected int tbpttBackLength = 20;
        protected InputType inputType;

        public Builder(MultiLayerNetwork origModel) {

            this.origModel = origModel;
            this.origConf = origModel.getLayerWiseConfigurations();
            this.origParams = origModel.params();

            this.inputPreProcessors = origConf.getInputPreProcessors();
            this.backpropType = origConf.getBackpropType();
            this.tbpttFwdLength = origConf.getTbpttFwdLength();
            this.tbpttBackLength = origConf.getTbpttBackLength();
            //this.inputType = new InputType.Type()?? //FIXME

            this.editedConfs.addAll(origConf.getConfs());
        }

        public Builder setTbpttFwdLength(int l) {
            this.tbpttFwdLength = l;
            return this;
        }

        public Builder setTbpttBackLength(int l) {
            this.tbpttBackLength = l;
            return this;
        }

        public Builder setFeatureExtractor(int layerNum) {
            this.frozenTill = layerNum;
            return this;
        }

        public Builder finetuneConfiguration(NeuralNetConfiguration.Builder newDefaultConfBuilder) {
            this.globalConfig = newDefaultConfBuilder;
            return this;
        }

        public Builder noutReplace(int layerNum, int nOut, WeightInit scheme) {
            nOutReplace(layerNum, nOut, scheme, scheme);
            return this;
        }

        public Builder nOutReplace(int layerNum, int nOut, WeightInit scheme, WeightInit schemeNext) {
            editedLayersMap.put(layerNum,new ImmutableTriple<>(nOut,scheme,schemeNext));
            return this;
        }

        public Builder popOutputLayer() {
            popFrom = origConf.getConfs().size()-1;
            return this;
        }

        public Builder popFromOutput(int layerNum) {
            if(popFrom == -1) {
                popFrom = layerNum;
            }
            else {
                throw new IllegalArgumentException("Pop from can only be called once");
            }
            return this;
        }

        public Builder addLayer(Layer layer) {

            if(!prepDone) {
                doPrep();
            }
            // Use the fineTune NeuralNetConfigurationBuilder and the layerConf to get the NeuralNetConfig
            //instantiate dummy layer to get the params
            editedConfs.add(globalConfig.clone().layer(layer).build());
            org.deeplearning4j.nn.api.Layer dummyLayer = editedConfs.get(editedConfs.size() - 1).getLayer().instantiate(globalConfig.build(),....)
            appendParams.add(dummyLayer.params());
            return this;
        }

        private void doPrep() {
            //first set finetune configs on all layers in model
            finetuneConfigurationBuild();
            //apply changes to nout/nin if any in sorted order
            if(!editedLayers.isEmpty()) {


            }
        }


        private void finetuneConfigurationBuild() {

            for (int i = 0; i < origConf.getConfs().size(); i++) {

                NeuralNetConfiguration layerConf = origConf.getConf(i);
                Layer layerConfImpl = layerConf.getLayer();

                //clear the learning related params for all layers in the origConf and set to defaults
                layerConfImpl.setUpdater(null);
                layerConfImpl.setMomentum(Double.NaN);
                layerConfImpl.setWeightInit(null);
                layerConfImpl.setBiasInit(Double.NaN);
                layerConfImpl.setDist(null);
                layerConfImpl.setLearningRate(Double.NaN);
                layerConfImpl.setBiasLearningRate(Double.NaN);
                layerConfImpl.setLearningRateSchedule(null);
                layerConfImpl.setMomentumSchedule(null);
                layerConfImpl.setL1(Double.NaN);
                layerConfImpl.setL2(Double.NaN);
                layerConfImpl.setDropOut(Double.NaN);
                layerConfImpl.setRho(Double.NaN);
                layerConfImpl.setEpsilon(Double.NaN);
                layerConfImpl.setRmsDecay(Double.NaN);
                layerConfImpl.setAdamMeanDecay(Double.NaN);
                layerConfImpl.setAdamVarDecay(Double.NaN);
                layerConfImpl.setGradientNormalization(GradientNormalization.None);
                layerConfImpl.setGradientNormalizationThreshold(1.0);

                editedConfs.set(i, globalConfig.clone().layer(layerConfImpl).build());
            }
        }

        private void nOutReplaceBuild(int layerNum, int nOut, WeightInit scheme, WeightInit schemeNext) {

            for (int i = 0; i < editedConfs.size(); i++) {
                NeuralNetConfiguration layerConf = editedConfs.get(i);
                Layer layerImpl = layerConf.getLayer();

                if (i < layerNum && tillIndex < 0) {
                    //first edit to architecture
                    tillIndex += layerImpl.initializer().numParams(layerConf);
                }
                else if (i == layerNum) {
                    layerImpl.setWeightInit(scheme);
                    String layerConfJson = layerConf.toJson();
                    NeuralNetConfiguration newLayerConf = layerConf.clone();
                    //FIXME: This is a hack via json to change nOut, Could expose nIn and nOut via setters instead?
                    ObjectMapper mapper = new ObjectMapper();
                    try {
                        JsonNode root = mapper.readTree(layerConfJson);
                        JsonNode layerRoot = root.path("layer").path(root.path("layer").fieldNames().next());
                        ((org.nd4j.shade.jackson.databind.node.ObjectNode) layerRoot).put("nout", nOut);
                        newLayerConf = mapper.readValue(root.toString(), NeuralNetConfiguration.class);
                        editedConfs.set(i, newLayerConf);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    int numParams = layerImpl.initializer().numParams(newLayerConf);
                    INDArray params = Nd4j.create(1, numParams);
                    org.deeplearning4j.nn.api.Layer someLayer = layerImpl.instantiate(layerConf, null, 0, params, true);
                    appendParams.add(someLayer.params());

                }
                else if (i == layerNum + 1) {
                    layerImpl.setWeightInit(schemeNext);
                    String layerConfJson = layerConf.toJson();
                    NeuralNetConfiguration newLayerConf = layerConf.clone();
                    //FIXME: This is a hack via json to change nIn, Could expose nIn and nOut via setters instead?
                    ObjectMapper mapper = new ObjectMapper();
                    try {
                        JsonNode root = mapper.readTree(layerConfJson);
                        JsonNode layerRoot = root.path("layer").path(root.path("layer").fieldNames().next());
                        ((org.nd4j.shade.jackson.databind.node.ObjectNode) layerRoot).put("nin", nOut);
                        newLayerConf = mapper.readValue(root.toString(), NeuralNetConfiguration.class);
                        editedConfs.set(i, newLayerConf);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    int numParams = layerImpl.initializer().numParams(newLayerConf);
                    INDArray params = Nd4j.create(1, numParams);
                    org.deeplearning4j.nn.api.Layer someLayer = layerImpl.instantiate(layerConf, null, 0, params, true);
                    appendParams.add(someLayer.params());
                }
                else {
                    org.deeplearning4j.nn.api.Layer origLayer = origModel.getLayer(i);
                    appendParams.add(origLayer.params());
                }
            }
        }

        public void popFromOutputToBuild(int layerNum) {
            for (int i = layerNum; i < origConf.getConfs().size(); i++) {
                editedConfs.remove(i);
            }
        }

        //unchecked?
        public MultiLayerNetwork build() {
            if (editConf() == null) {
                editedModel = origModel;
            } else {
                editedModel = new MultiLayerNetwork(editConf(), constructParams());
            }

            org.deeplearning4j.nn.api.Layer[] layers = editedModel.getLayers();
            for (int i = frozenTill; i >= 0; i--) {
                layers[i] = new FrozenLayer(layers[i]);
            }
            editedModel.setLayers(layers);
            return editedModel;
        }

        private INDArray constructParams() {
            INDArray keepView = origParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, tillIndex, true));
            INDArray appendView = Nd4j.hstack(appendParams);
            return Nd4j.hstack(keepView, appendView);
        }

        private MultiLayerConfiguration editConf() {
            //use the editedConfs list to make a new config
            return new MultiLayerConfiguration.Builder().backprop(backprop).inputPreProcessors(inputPreProcessors).
                    pretrain(pretrain).backpropType(backpropType).tBPTTForwardLength(tbpttFwdLength)
                    .tBPTTBackwardLength(tbpttBackLength)
                    .setInputType(this.inputType)
                    .confs(editedConfs).build();
        }
    }
}
