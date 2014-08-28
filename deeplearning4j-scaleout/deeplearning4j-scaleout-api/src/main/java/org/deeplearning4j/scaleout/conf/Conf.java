package org.deeplearning4j.scaleout.conf;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang3.SerializationUtils;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.linalg.api.activation.ActivationFunction;
import org.deeplearning4j.linalg.api.activation.Activations;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.transformation.MatrixTransform;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.api.NeuralNetwork.OptimizationAlgorithm;

import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;


/**
 * Conf used for distributed deep learning
 * @author Adam Gibson
 *
 */
public class Conf implements Serializable,Cloneable {


    private static final long serialVersionUID = 2994146097289344262L;
    private Class<? extends BaseMultiLayerNetwork> multiLayerClazz;
    private Class<? extends NeuralNetwork> neuralNetworkClazz;
    private int k;
    private long seed = 123;
    private float corruptionLevel = 0.3f;
    private float sparsity = 0;
    private ActivationFunction function = Activations.sigmoid();
    private ActivationFunction outputActivationFunction = Activations.softmax();
    private int[] layerSizes = new int[]{300,300,300};
    private int pretrainEpochs = 1000;
    private int finetuneEpochs = 1000;
    private float pretrainLearningRate = 0.01f;
    private float finetuneLearningRate = 0.01f;
    private int split = 10;
    private int nIn = 1;
    private int nOut = 1;
    private int numPasses = 1;
    private float momentum = 0.1f;
    private boolean useRegularization = false;
    private Object[] deepLearningParams;
    private String masterUrl;
    private float l2;
    private Map<Integer,MatrixTransform> weightTransforms = new HashMap<>();
    private Map<Integer,ActivationFunction> activationFunctionForLayer = new HashMap<>();
    private Map<Integer,Integer> renderEpochsByLayer = new HashMap<>();
    private int renderWeightEpochs = -1;
    private String masterAbsPath;
    private INDArray columnMeans;
    private INDArray columnStds;
    private boolean useAdaGrad = false;
    private boolean useBackProp = true;
    private float dropOut;
    private OptimizationAlgorithm optimizationAlgorithm = OptimizationAlgorithm.CONJUGATE_GRADIENT;
    private boolean normalizeZeroMeanAndUnitVariance;
    private boolean scale;
    private RBM.VisibleUnit visibleUnit = RBM.VisibleUnit.BINARY;
    private RBM.HiddenUnit hiddenUnit = RBM.HiddenUnit.BINARY;
    private String stateTrackerConnectionString;
    private Map<Integer,RBM.VisibleUnit> visibleUnitByLayer = new HashMap<>();
    private Map<Integer,RBM.HiddenUnit> hiddenUnitByLayer = new HashMap<>();
    private Map<Integer,Float> learningRateForLayer = new HashMap<>();
    private boolean roundCodeLayer = false;
    private boolean normalizeCodeLayer = false;
    private boolean lineSearchBackProp = false;
    private boolean sampleHiddenActivations = false;
    private Map<Integer,Boolean> sampleHiddenActivationsByLayer = new HashMap<>();
    private boolean useDropConnect = false;
    private float outputLayerDropOut = 0.0f;
    private NeuralNetConfiguration conf;

    public NeuralNetConfiguration getConf() {
        return conf;
    }

    public void setConf(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    public boolean isUseDropConnect() {
        return useDropConnect;
    }

    public void setUseDropConnect(boolean useDropConnect) {
        this.useDropConnect = useDropConnect;
    }

    public float getOutputLayerDropOut() {
        return outputLayerDropOut;
    }

    public void setOutputLayerDropOut(float outputLayerDropOut) {
        this.outputLayerDropOut = outputLayerDropOut;
    }

    public Map<Integer, Boolean> getSampleHiddenActivationsByLayer() {
        return sampleHiddenActivationsByLayer;
    }

    public void setSampleHiddenActivationsByLayer(Map<Integer, Boolean> sampleHiddenActivationsByLayer) {
        this.sampleHiddenActivationsByLayer = sampleHiddenActivationsByLayer;
    }

    public boolean isSampleHiddenActivations() {
        return sampleHiddenActivations;
    }

    public void setSampleHiddenActivations(boolean sampleHiddenActivations) {
        this.sampleHiddenActivations = sampleHiddenActivations;
    }

    public boolean isLineSearchBackProp() {
        return lineSearchBackProp;
    }

    public void setLineSearchBackProp(boolean lineSearchBackProp) {
        this.lineSearchBackProp = lineSearchBackProp;
    }

    public Map<Integer, Integer> getRenderEpochsByLayer() {
        return renderEpochsByLayer;
    }

    public void setRenderEpochsByLayer(Map<Integer, Integer> renderEpochsByLayer) {
        this.renderEpochsByLayer = renderEpochsByLayer;
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

    public Map<Integer, Float> getLearningRateForLayer() {
        return learningRateForLayer;
    }



    public void setLearningRateForLayer(Map<Integer, Float> learningRateForLayer) {
        this.learningRateForLayer = learningRateForLayer;
    }

    public ActivationFunction getOutputActivationFunction() {
        return outputActivationFunction;
    }

    public void setOutputActivationFunction(ActivationFunction outputActivationFunction) {
        this.outputActivationFunction = outputActivationFunction;
    }

    public Map<Integer, RBM.VisibleUnit> getVisibleUnitByLayer() {
        return visibleUnitByLayer;
    }

    public void setVisibleUnitByLayer(Map<Integer, RBM.VisibleUnit> visibleUnitByLayer) {
        this.visibleUnitByLayer = visibleUnitByLayer;
    }

    public Map<Integer, RBM.HiddenUnit> getHiddenUnitByLayer() {
        return hiddenUnitByLayer;
    }

    public void setHiddenUnitByLayer(Map<Integer, RBM.HiddenUnit> hiddenUnitByLayer) {
        this.hiddenUnitByLayer = hiddenUnitByLayer;
    }

    public Map<Integer, ActivationFunction> getActivationFunctionForLayer() {
        return activationFunctionForLayer;
    }

    public void setActivationFunctionForLayer(Map<Integer, ActivationFunction> activationFunctionForLayer) {
        this.activationFunctionForLayer = activationFunctionForLayer;
    }

    public String getStateTrackerConnectionString() {
        return stateTrackerConnectionString;
    }

    public void setStateTrackerConnectionString(String stateTrackerConnectionString) {
        this.stateTrackerConnectionString = stateTrackerConnectionString;
    }

    public RBM.VisibleUnit getVisibleUnit() {
        return visibleUnit;
    }

    public void setVisibleUnit(RBM.VisibleUnit visibleUnit) {
        this.visibleUnit = visibleUnit;
    }

    public RBM.HiddenUnit getHiddenUnit() {
        return hiddenUnit;
    }

    public void setHiddenUnit(RBM.HiddenUnit hiddenUnit) {
        this.hiddenUnit = hiddenUnit;
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

    public float getDropOut() {
        return dropOut;
    }


    public void setDropOut(float dropOut) {
        this.dropOut = dropOut;
    }


    /**
     * Sets in and outs based on data
     * @param data the data to use
     */
    public void initFromData(DataSet data) {
        setnIn(data.numInputs());
        setnOut(data.numOutcomes());
    }



    public synchronized boolean isUseAdaGrad() {
        return useAdaGrad;
    }
    public synchronized void setUseAdaGrad(boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
    }
    public synchronized String getMasterAbsPath() {
        return masterAbsPath;
    }
    public synchronized void setMasterAbsPath(String masterAbsPath) {
        this.masterAbsPath = masterAbsPath;
    }
    public synchronized float getSparsity() {
        return sparsity;
    }
    public synchronized void setSparsity(float sparsity) {
        this.sparsity = sparsity;
    }
    public Map<Integer, MatrixTransform> getWeightTransforms() {
        return weightTransforms;
    }
    public void setWeightTransforms(Map<Integer, MatrixTransform> weightTransforms) {
        this.weightTransforms = weightTransforms;
    }
    public float getL2() {
        return l2;
    }
    public void setL2(float l2) {
        this.l2 = l2;
    }
    public String getMasterUrl() {
        return masterUrl;
    }
    public void setMasterUrl(String masterUrl) {
        this.masterUrl = masterUrl;
    }
    public float getMomentum() {
        return momentum;
    }
    public void setMomentum(float momentum) {
        this.momentum = momentum;
    }
    public boolean isUseRegularization() {
        return useRegularization;
    }
    public void setUseRegularization(boolean useRegularization) {
        this.useRegularization = useRegularization;
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
    public int getK() {
        return k;
    }
    public void setK(int k) {
        this.k = k;
    }
    public long getSeed() {
        return seed;
    }
    public void setSeed(long seed) {
        this.seed = seed;
    }
    public float getCorruptionLevel() {
        return corruptionLevel;
    }
    public void setCorruptionLevel(float corruptionLevel) {
        this.corruptionLevel = corruptionLevel;
    }
    public ActivationFunction getFunction() {
        return function;
    }
    public void setFunction(ActivationFunction function) {
        this.function = function;
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



    public synchronized INDArray getColumnMeans() {
        return columnMeans;
    }
    public synchronized void setColumnMeans(INDArray columnMeans) {
        this.columnMeans = columnMeans;
    }
    public synchronized INDArray getColumnStds() {
        return columnStds;
    }
    public synchronized void setColumnStds(INDArray columnStds) {
        this.columnStds = columnStds;
    }
    public void setLayerSizes(Integer[] layerSizes) {
        this.layerSizes = new int[layerSizes.length];
        for(int i = 0; i < layerSizes.length; i++)
            this.layerSizes[i] = layerSizes[i];
    }

    public int getPretrainEpochs() {
        return pretrainEpochs;
    }
    public void setPretrainEpochs(int pretrainEpochs) {
        this.pretrainEpochs = pretrainEpochs;
    }
    public float getPretrainLearningRate() {
        return pretrainLearningRate;
    }

    /**
     * Sets the pretrain learning rate.
     * Note that this will also be used for adagrad
     * pretrain master learning rate
     * @param pretrainLearningRate the learning rate to use
     */
    public void setPretrainLearningRate(float pretrainLearningRate) {
        this.pretrainLearningRate = pretrainLearningRate;
    }
    public float getFinetuneLearningRate() {
        return finetuneLearningRate;
    }



    /**
     * Sets the finetune learning rate.
     * Note that this will also be used for adagrad
     * finetune master learning rate
     * @param finetuneLearningRate the learning rate to use
     */
    public void setFinetuneLearningRate(float finetuneLearningRate) {
        this.finetuneLearningRate = finetuneLearningRate;
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
    public int getnIn() {
        return nIn;
    }
    public void setnIn(int nIn) {
        this.nIn = nIn;
    }
    public int getnOut() {
        return nOut;
    }
    public void setnOut(int nOut) {
        this.nOut = nOut;
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


    public int getFinetuneEpochs() {
        return finetuneEpochs;
    }
    public void setFinetuneEpochs(int finetuneEpochs) {
        this.finetuneEpochs = finetuneEpochs;
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



    public OptimizationAlgorithm getOptimizationAlgorithm() {
        return optimizationAlgorithm;
    }


    public void setOptimizationAlgorithm(OptimizationAlgorithm optimizationAlgorithm) {
        this.optimizationAlgorithm = optimizationAlgorithm;
    }


    /**
     * Returns a multi layer network based on the configuration
     * @return the initialized network
     */
    public BaseMultiLayerNetwork init() {
        if(getMultiLayerClazz().isAssignableFrom(DBN.class)) {
            return new DBN.Builder().configure(conf)
            .withClazz(getMultiLayerClazz()).lineSearchBackProp(isLineSearchBackProp())
                    .hiddenLayerSizes(getLayerSizes())
                    .useDropConnection(isUseDropConnect())
                    .build();



        }

        else {
            return  new BaseMultiLayerNetwork.Builder<>().withClazz(getMultiLayerClazz())
                    .hiddenLayerSizes(getLayerSizes())
                     .lineSearchBackProp(isLineSearchBackProp()) .
                     useDropConnection(isUseDropConnect())

                    .build();

        }
    }


}
