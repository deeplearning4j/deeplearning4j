package org.deeplearning4j.scaleout.conf;

import org.apache.commons.lang3.SerializationUtils;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.rng.SynchronizedRandomGenerator;
import org.nd4j.linalg.transformation.MatrixTransform;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Conf used for distributed deep learning
 * @author Adam Gibson
 *
 */
public class Conf implements Serializable,Cloneable {

    private static final long serialVersionUID = 2994146097289344262L;
    private Class<? extends BaseMultiLayerNetwork> multiLayerClazz;
    private Class<? extends NeuralNetwork> neuralNetworkClazz;
    private int[] layerSizes = new int[]{300,300,300};

    private int split = 10;
    private int numPasses = 1;
    private Object[] deepLearningParams;
    private String masterUrl;
    private Map<Integer,MatrixTransform> weightTransforms = new HashMap<>();
    private int renderWeightEpochs = -1;
    private String masterAbsPath;

    private boolean useBackProp = true;
    private boolean normalizeZeroMeanAndUnitVariance;
    private boolean scale;

    private String stateTrackerConnectionString;

    private boolean roundCodeLayer = false;
    private boolean normalizeCodeLayer = false;
    private boolean lineSearchBackProp = false;
    private NeuralNetConfiguration conf;
    private List<NeuralNetConfiguration> layerConfigs = new ArrayList<>();

    public NeuralNetConfiguration getConf() {
        return conf;
    }

    public void setConf(NeuralNetConfiguration conf) {
        if(conf.getRng() != null && !(conf.getRng() instanceof SynchronizedRandomGenerator)) {
            conf.setRng(new SynchronizedRandomGenerator(conf.getRng()));
        }
        this.conf = conf;
    }
    public void setLayerConfigs() {
        List<NeuralNetConfiguration> layers = new ArrayList<>();
        for(int i = 0; i < layerSizes.length; i++) {
            layers.add(conf.clone());
        }
        setLayerConfigs(layers);
    }
    public void setLayerConfigs(List<NeuralNetConfiguration> l) { this.layerConfigs = l; }

    public boolean isLineSearchBackProp() {
        return lineSearchBackProp;
    }

    public void setLineSearchBackProp(boolean lineSearchBackProp) {
        this.lineSearchBackProp = lineSearchBackProp;
    }

    public boolean isNormalizeCodeLayer() {
        return normalizeCodeLayer;
    }

    public void setNormalizeCodeLayer(boolean normalizeCodeLayer) {
        this.normalizeCodeLayer = normalizeCodeLayer;
    }

    public boolean isRoundCodeLayer() {
        return roundCodeLayer;
    }

    public void setRoundCodeLayer(boolean roundCodeLayer) {
        this.roundCodeLayer = roundCodeLayer;
    }

    public String getStateTrackerConnectionString() {
        return stateTrackerConnectionString;
    }

    public void setStateTrackerConnectionString(String stateTrackerConnectionString) {
        this.stateTrackerConnectionString = stateTrackerConnectionString;
    }

    public boolean isScale() {
        return scale;
    }

    public void setScale(boolean scale) {
        this.scale = scale;
    }

    public boolean isNormalizeZeroMeanAndUnitVariance() {
        return normalizeZeroMeanAndUnitVariance;
    }

    public void setNormalizeZeroMeanAndUnitVariance(boolean normalizeZeroMeanAndUnitVariance) {
        this.normalizeZeroMeanAndUnitVariance = normalizeZeroMeanAndUnitVariance;
    }

    public synchronized String getMasterAbsPath() {
        return masterAbsPath;
    }
    public synchronized void setMasterAbsPath(String masterAbsPath) {
        this.masterAbsPath = masterAbsPath;
    }
    public Map<Integer, MatrixTransform> getWeightTransforms() {
        return weightTransforms;
    }
    public void setWeightTransforms(Map<Integer, MatrixTransform> weightTransforms) {
        this.weightTransforms = weightTransforms;
    }

    public String getMasterUrl() {
        return masterUrl;
    }
    public void setMasterUrl(String masterUrl) {
        this.masterUrl = masterUrl;
    }
    public Class<? extends BaseMultiLayerNetwork> getMultiLayerClazz() {
        return multiLayerClazz;
    }
    public void setMultiLayerClazz(
            Class<? extends BaseMultiLayerNetwork> multiLayerClazz) {
        this.multiLayerClazz = multiLayerClazz;
    }
    public Class<? extends NeuralNetwork> getNeuralNetworkClazz() {
        return neuralNetworkClazz;
    }
    public void setNeuralNetworkClazz(
            Class<? extends NeuralNetwork> neuralNetworkClazz) {
        this.neuralNetworkClazz = neuralNetworkClazz;
    }

    /**
     * Returns the hidden layer sizes
     * @return the hidden layer sizes applyTransformToDestination for this configuration
     */
    public int[] getLayerSizes() {
        return layerSizes;
    }

    /**
     * Sets the hidden layer sizes
     * @param layerSizes the layer sizes to use
     */
    public void setLayerSizes(int[] layerSizes) {
        this.layerSizes = layerSizes;
    }



   public void setLayerSizes(Integer[] layerSizes) {
        this.layerSizes = new int[layerSizes.length];
        for(int i = 0; i < layerSizes.length; i++)
            this.layerSizes[i] = layerSizes[i];
    }


    public int getSplit() {
        return split;
    }

    /**
     * The optimal split is usually going to be something akin to
     * the number of workers * the mini batch size.
     *
     * Say if you have a system with 8 cores with 1 core per worker
     * and a batch size of 10, you will want 80 as the batch size.
     * @param split
     */
    public void setSplit(int split) {
        this.split = split;
    }

    /**
     * Number of epochs to run
     * @return the number of epochs to run
     */
    public int getNumPasses() {
        return numPasses;
    }
    public void setNumPasses(int numPasses) {
        this.numPasses = numPasses;
    }
    public Object[] getDeepLearningParams() {
        return deepLearningParams;
    }
    public void setDeepLearningParams(Object[] deepLearningParams) {
        this.deepLearningParams = deepLearningParams;
    }

    public int getRenderWeightEpochs() {
        return renderWeightEpochs;
    }
    public void setRenderWeightEpochs(int renderWeightEpochs) {
        this.renderWeightEpochs = renderWeightEpochs;
    }
    public Conf copy() {
        return SerializationUtils.clone(this);
    }

    public  boolean isUseBackProp() {
        return useBackProp;
    }


    public  void setUseBackProp(boolean useBackProp) {
        this.useBackProp = useBackProp;
    }


    /**
     * Corruption level of 0.3 and learning rate of 0.01
     * and 1000 epochs
     * @return
     */
    public static Object[] getDefaultDenoisingAutoEncoderParams() {
        return new Object[]{0.3,0.01,1000};
    }
    /**
     * K of 1 and learning rate of 0.01 and 1000 epochs
     * @return the default parameters for RBMs
     * and DBNs
     */
    public static Object[] getDefaultRbmParams() {
        return new Object[]{1,0.01,1000};
    }

    /**
     * Returns a multi layer network based on the configuration
     * @return the initialized network
     */
    public BaseMultiLayerNetwork init() {
        if(getMultiLayerClazz().isAssignableFrom(DBN.class)) {
            return new DBN.Builder().configure(conf).layerWiseCOnfiguration(layerConfigs)
            .withClazz(getMultiLayerClazz()).lineSearchBackProp(isLineSearchBackProp())
                    .hiddenLayerSizes(getLayerSizes())
                    .build();



        }

        else {
            return  new BaseMultiLayerNetwork.Builder<>().withClazz(getMultiLayerClazz())
                    .hiddenLayerSizes(getLayerSizes())
                     .lineSearchBackProp(isLineSearchBackProp())
                    .build();

        }
    }


}
