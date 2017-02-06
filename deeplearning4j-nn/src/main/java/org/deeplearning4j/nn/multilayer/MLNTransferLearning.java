package org.deeplearning4j.nn.multilayer;

import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Triple;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;


/**
 * Other things to consider:
 * - There really should be a way to featurize and save to disk and then train from the featurized data. This will help users iterate quicker and
 * get a "nano net" that converges fast and then they can "fineTune" to their heart's content without wondering about having disruptive gradients
 * flowing backward to the unfrozen layers.
 * - And then adapting this for computation graphs (yikes)
 * - Also a summary of the model before and after to show how many new params were added/deleted and how many are learnable and how many are frozen etc..
 */
public class MLNTransferLearning {

    private INDArray origParams;
    private MultiLayerConfiguration origConf;
    private MultiLayerNetwork origModel;

    private MultiLayerNetwork editedModel;
    private NeuralNetConfiguration.Builder globalConfig;
    private int frozenTill = -1;
    private int popN = 0;
    private boolean prepDone = false;
    private List<Integer> editedLayers = new ArrayList<>();
    private Map<Integer, Triple<Integer, WeightInit, WeightInit>> editedLayersMap = new HashMap<>();
    private List<INDArray> editedParams = new ArrayList<>();
    private List<NeuralNetConfiguration> editedConfs = new ArrayList<>();
    private List<INDArray> appendParams = new ArrayList<>(); //these could be new arrays, and views from origParams
    private List<NeuralNetConfiguration> appendConfs = new ArrayList<>();

    private Map<Integer, InputPreProcessor> inputPreProcessors = new HashMap<>();
    private boolean pretrain = false;
    private boolean backprop = true;
    private BackpropType backpropType = BackpropType.Standard;
    private int tbpttFwdLength = 20;
    private int tbpttBackLength = 20;
    private InputType inputType;

    public MLNTransferLearning(MultiLayerNetwork origModel) {

        this.origModel = origModel;
        this.origConf = origModel.getLayerWiseConfigurations();
        this.origParams = origModel.params();

        this.inputPreProcessors = origConf.getInputPreProcessors();
        this.backpropType = origConf.getBackpropType();
        this.tbpttFwdLength = origConf.getTbpttFwdLength();
        this.tbpttBackLength = origConf.getTbpttBackLength();
        //this.inputType = new InputType.Type()?? //FIXME
    }

    public MLNTransferLearning setTbpttFwdLength(int l) {
        this.tbpttFwdLength = l;
        return this;
    }

    public MLNTransferLearning setTbpttBackLength(int l) {
        this.tbpttBackLength = l;
        return this;
    }

    public MLNTransferLearning setFeatureExtractor(int layerNum) {
        this.frozenTill = layerNum;
        return this;
    }

    /**
     * NeuralNetConfiguration builder to set options (learning rate, updater etc..) for learning
     * Note that this will clear and override all other learning related settings in non frozen layers
     *
     * @param newDefaultConfBuilder
     * @return
     */
    public MLNTransferLearning fineTuneConfiguration(NeuralNetConfiguration.Builder newDefaultConfBuilder) {
        this.globalConfig = newDefaultConfBuilder;
        return this;
    }

    /**
     * Modify the architecture of a layer by changing nOut
     * Note this will also affect the layer that follows the layer specified, unless it is the output layer
     *
     * @param layerNum The index of the layer to change nOut of
     * @param nOut     Value of nOut to change to
     * @param scheme   Weight Init scheme to use for params
     * @return
     */
    public MLNTransferLearning nOutReplace(int layerNum, int nOut, WeightInit scheme) {
        editedLayers.add(layerNum);
        editedLayersMap.put(layerNum, new ImmutableTriple<>(nOut, scheme, scheme));
        return this;
    }

    /**
     * Modify the architecture of a layer by changing nOut
     * Note this will also affect the layer that follows the layer specified, unless it is the output layer
     * Can specify different weight init schemes for the specified layer and the layer that follows it.
     *
     * @param layerNum   The index of the layer to change nOut of
     * @param nOut       Value of nOut to change to
     * @param scheme     Weight Init scheme to use for params in the layerNum
     * @param schemeNext Weight Init scheme to use for params in the layerNum+1
     * @return
     */
    public MLNTransferLearning nOutReplace(int layerNum, int nOut, WeightInit scheme, WeightInit schemeNext) {
        editedLayers.add(layerNum);
        editedLayersMap.put(layerNum, new ImmutableTriple<>(nOut, scheme, schemeNext));
        return this;
    }

    /**
     * Helper method to remove the outputLayer of the net.
     * Only one of the two - popOutputLayer() or popFromOutput(layerNum) - can be specified
     * When layers are popped at the very least an output layer should be added with .addLayer(...)
     *
     * @return
     */
    public MLNTransferLearning popOutputLayer() {
        popN = 1;
        return this;
    }

    /**
     * Pop last "n" layers of the net
     *
     * @param layerNum number of layers to pop, 1 will pop output layer only and so on...
     * @return
     */
    public MLNTransferLearning popFromOutput(int layerNum) {
        if (popN == 0) {
            popN = layerNum;
        } else {
            throw new IllegalArgumentException("Pop from can only be called once");
        }
        return this;
    }

    /**
     * Add layers to the net
     * Required if layers are popped. Can be called multiple times and layers will be added in the order with which they were called.
     * At the very least an outputLayer must be added (output layer should be added last - as per the note on order)
     * Learning configs like updaters, learning rate etc specified per layer, here will be honored
     *
     * @param layer layer conf to add
     * @return
     */
    public MLNTransferLearning addLayer(Layer layer) {

        if (!prepDone) {
            doPrep();
        }
        // Use the fineTune NeuralNetConfigurationBuilder and the layerConf to get the NeuralNetConfig
        //instantiate dummy layer to get the params
        NeuralNetConfiguration layerConf = globalConfig.clone().layer(layer).build();
        Layer layerImpl = layerConf.getLayer();
        int numParams = layerImpl.initializer().numParams(layerConf);
        INDArray params;
        if (numParams > 0) {
            params = Nd4j.create(1, numParams);
            org.deeplearning4j.nn.api.Layer someLayer = layerImpl.instantiate(layerConf, null, 0, params, true);
            appendParams.add(someLayer.params());
            appendConfs.add(someLayer.conf());
        }
        else {
            appendConfs.add(layerConf);

        }
        return this;
    }

    /**
     * Specify the preprocessor for the added layers
     * for cases where they cannot be inferred automatically.
     * @param index of the layer
     * @param processor to be used on the data
     * @return
     */
    public MLNTransferLearning setInputPreProcessor(Integer layer, InputPreProcessor processor) {
        inputPreProcessors.put(layer,processor);
        return this;
    }

    /**
     * Returns a model with the fine tune configuration and specified architecture changes.
     * .init() need not be called. Can be directly fit.
     *
     * @return
     */
    public MultiLayerNetwork build() {

        if (!prepDone) {
            doPrep();
        }

        editedModel = new MultiLayerNetwork(constructConf(), constructParams());
        if (frozenTill != -1) {
            org.deeplearning4j.nn.api.Layer[] layers = editedModel.getLayers();
            for (int i = frozenTill; i >= 0; i--) {
                //unchecked?
                layers[i] = new FrozenLayer(layers[i]);
            }
            editedModel.setLayers(layers);
        }
        return editedModel;
    }

    private void doPrep() {

        if (globalConfig == null) {
            throw new IllegalArgumentException("FineTrain config must be set with .fineTuneConfiguration");
        }

        //first set finetune configs on all layers in model
        fineTuneConfigurationBuild();

        //editParams gets original model params
        for (int i = 0; i < origModel.getnLayers(); i++) {
            editedParams.add(origModel.getLayer(i).params());
        }
        //apply changes to nout/nin if any in sorted order and save to editedParams
        if (!editedLayers.isEmpty()) {
            Integer[] editedLayersSorted = editedLayers.toArray(new Integer[editedLayers.size()]);
            Arrays.sort(editedLayersSorted);
            for (int i = 0; i < editedLayersSorted.length; i++) {
                int layerNum = editedLayersSorted[i];
                nOutReplaceBuild(layerNum, editedLayersMap.get(layerNum).getLeft(), editedLayersMap.get(layerNum).getMiddle(), editedLayersMap.get(layerNum).getRight());
            }
        }

        //finally pop layers specified
        int i = 0;
        while (i < popN) {
            Integer layerNum = origModel.getnLayers() - i;
            if (inputPreProcessors.containsKey(layerNum)) {
                inputPreProcessors.remove(layerNum);
            }
            editedConfs.remove(editedConfs.size() - 1);
            editedParams.remove(editedParams.size() - 1);
            i++;
        }
        prepDone = true;

    }


    private void fineTuneConfigurationBuild() {

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

            editedConfs.add(globalConfig.clone().layer(layerConfImpl).build());
        }
    }

    private void nOutReplaceBuild(int layerNum, int nOut, WeightInit scheme, WeightInit schemeNext) {

        NeuralNetConfiguration layerConf = editedConfs.get(layerNum);
        Layer layerImpl = layerConf.getLayer();
        layerImpl.setWeightInit(scheme);
        FeedForwardLayer layerImplF = (FeedForwardLayer) layerImpl;
        layerImplF.overrideNOut(nOut, true);
        int numParams = layerImpl.initializer().numParams(layerConf);
        INDArray params = Nd4j.create(1, numParams);
        org.deeplearning4j.nn.api.Layer someLayer = layerImpl.instantiate(layerConf, null, 0, params, true);
        editedParams.set(layerNum, someLayer.params());

        if (layerNum + 1 < editedConfs.size()) {
            layerConf = editedConfs.get(layerNum + 1);
            layerImpl = layerConf.getLayer();
            layerImpl.setWeightInit(schemeNext);
            layerImplF = (FeedForwardLayer) layerImpl;
            layerImplF.overrideNIn(nOut, true);
            numParams = layerImpl.initializer().numParams(layerConf);
            params = Nd4j.create(1, numParams);
            someLayer = layerImpl.instantiate(layerConf, null, 0, params, true);
            editedParams.set(layerNum + 1, someLayer.params());
        }

    }

    private INDArray constructParams() {
        INDArray keepView = Nd4j.hstack(editedParams);
        if (!appendParams.isEmpty()) {
            INDArray appendView = Nd4j.hstack(appendParams);
            return Nd4j.hstack(keepView, appendView);
        } else {
            return keepView;
        }
    }

    private MultiLayerConfiguration constructConf() {
        //use the editedConfs list to make a new config
        List<NeuralNetConfiguration> allConfs = new ArrayList<>();
        allConfs.addAll(editedConfs);
        allConfs.addAll(appendConfs);
        return new MultiLayerConfiguration.Builder().backprop(backprop).inputPreProcessors(inputPreProcessors).
                pretrain(pretrain).backpropType(backpropType).tBPTTForwardLength(tbpttFwdLength)
                .tBPTTBackwardLength(tbpttBackLength)
                .setInputType(this.inputType)
                .confs(allConfs).build();
    }
}
