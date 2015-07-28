/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;

import lombok.Data;
import lombok.NoArgsConstructor;

import org.deeplearning4j.nn.conf.deserializers.*;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.serializers.*;
import org.deeplearning4j.nn.conf.stepfunctions.NegativeDefaultStepFunction;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.conf.stepfunctions.GradientStepFunction;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.rng.DefaultRandom;
import org.deeplearning4j.util.Dl4jReflection;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.util.ClassUtils;

import java.io.IOException;
import java.io.Serializable;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * A Serializable configuration
 * for neural nets that covers per layer parameters
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class NeuralNetConfiguration implements Serializable,Cloneable {

    private double sparsity = 0;
    @Deprecated
    private boolean useAdaGrad = true;
    private double lr = 1e-1;
    protected double corruptionLevel = 0.3;
    protected int numIterations = 5;
    /* momentum for learning */
    protected double momentum = 0.5;
    /* L2 Regularization constant */
    protected double l2 = 0;
    protected boolean useRegularization = false;
    protected Updater updater = Updater.ADAGRAD;
    private String customLossFunction;
    //momentum after n iterations
    protected Map<Integer,Double> momentumAfter = new HashMap<>();
    //reset adagrad historical gradient after n iterations
    protected int resetAdaGradIterations = -1;
    //number of line search iterations
    @Deprecated
    protected int numLineSearchIterations = 5;
    protected int maxNumLineSearchIterations = 5;
    protected double dropOut = 0;
    //use only when binary hidden neuralNets are active
    protected boolean applySparsity = false;
    //weight init scheme, this can either be a distribution or a applyTransformToDestination scheme
    protected WeightInit weightInit = WeightInit.XAVIER;
    protected OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.CONJUGATE_GRADIENT;
    public LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY;
    //whether to constrain the gradient to unit norm or not
    protected boolean constrainGradientToUnitNorm = false;
    /* RNG for sampling. */
    @Deprecated
    protected transient DefaultRandom rng;
    //adadelta - weight for how much to consider previous history
    protected double rho;
    protected long seed;
    //weight initialization
    protected Distribution dist;
    protected StepFunction stepFunction = new GradientStepFunction();
    protected Layer layer;



    //gradient keys used for ensuring order when getting and setting the gradient
    protected List<String> variables = new ArrayList<>();
    //feed forward nets
    protected int nIn,nOut;

    protected String activationFunction;

    protected boolean useDropConnect = false;

    //RBMs
    private RBM.VisibleUnit visibleUnit = RBM.VisibleUnit.BINARY;
    private RBM.HiddenUnit hiddenUnit = RBM.HiddenUnit.BINARY;
    protected int k = 1;

    private int[] weightShape;

    //convolutional nets: this is the height and width of the kernel
    private int[] kernelSize = {2,2};
    //aka pool size for subsampling
    private int[] stride = {2,2};
    //kernel size for a convolutional net
    @Deprecated
    protected int kernel = 5;
    //batch size: primarily used for conv nets. Will be reinforced if set.
    protected int batchSize = 10;
    //minimize or maximize objective
    protected boolean minimize = false;
    //l1 regularization
    protected double l1 = 0.0;
    //feature map
    protected int[] featureMapSize = {9,9};

    protected double rmsDecay = 0.0;


    protected boolean miniBatch = false;


    protected Convolution.Type convolutionType = Convolution.Type.VALID;
    protected SubsamplingLayer.poolingType poolingType = SubsamplingLayer.poolingType.MAX;


    public NeuralNetConfiguration(double sparsity,
                                  boolean useAdaGrad,
                                  double lr,
                                  int k,
                                  double corruptionLevel,
                                  int numIterations,
                                  double momentum,
                                  double l2,
                                  boolean useRegularization,
                                  Map<Integer, Double> momentumAfter,
                                  int resetAdaGradIterations,
                                  double dropOut,
                                  boolean applySparsity,
                                  WeightInit weightInit,
                                  OptimizationAlgorithm optimizationAlgo,
                                  LossFunctions.LossFunction lossFunction,
                                  boolean constrainGradientToUnitNorm,
                                  DefaultRandom rng,
                                  long seed,
                                  Distribution dist,
                                  int nIn,
                                  int nOut,
                                  String activationFunction,
                                  RBM.VisibleUnit visibleUnit,
                                  RBM.HiddenUnit hiddenUnit,
                                  int[] weightShape,
                                  int[] filterSize,
                                  int filterDepth,
                                  int[] stride,
                                  int[] featureMapSize,
                                  int kernel,
                                  int batchSize,
                                  int numLineSearchIterations,
                                  int maxNumLineSearchIterations,
                                  boolean minimize,
                                  Layer layer, Convolution.Type convolutionType,
                                  SubsamplingLayer.poolingType poolingType,
                                  double l1,String customLossFunction) {
        this.minimize = minimize;
        this.customLossFunction = customLossFunction;
        this.convolutionType = convolutionType;
        this.poolingType = poolingType;
        this.numLineSearchIterations = numLineSearchIterations;
        this.maxNumLineSearchIterations = maxNumLineSearchIterations;
        this.featureMapSize = featureMapSize;
        this.l1 = l1;
        this.batchSize = batchSize;
        this.layer = layer;
        this.sparsity = sparsity;
        this.useAdaGrad = useAdaGrad;
        this.lr = lr;
        this.kernel = kernel;
        this.k = k;
        this.corruptionLevel = corruptionLevel;
        this.numIterations = numIterations;
        this.momentum = momentum;
        this.l2 = l2;
        this.useRegularization = useRegularization;
        this.momentumAfter = momentumAfter;
        this.resetAdaGradIterations = resetAdaGradIterations;
        this.dropOut = dropOut;
        this.applySparsity = applySparsity;
        this.weightInit = weightInit;
        this.optimizationAlgo = optimizationAlgo;
        this.lossFunction = lossFunction;
        this.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
        this.rng = null;
        this.seed = seed;
        this.dist = dist;
        this.nIn = nIn;
        this.nOut = nOut;
        this.activationFunction = activationFunction;
        this.visibleUnit = visibleUnit;
        this.hiddenUnit = hiddenUnit;
        if(weightShape != null)
            this.weightShape = weightShape;
        else
            this.weightShape = new int[]{nIn,nOut};
        this.kernelSize = filterSize;
        this.stride = stride;
    }

    /**
     * Copy constructor
     * @param neuralNetConfiguration
     */
    public NeuralNetConfiguration(NeuralNetConfiguration neuralNetConfiguration) {
        Field[] fields = NeuralNetConfiguration.class.getDeclaredFields();
        for(Field f : fields) {
            try {
                f.setAccessible(true);
                f.set(this,f.get(neuralNetConfiguration));
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }
        }

        if(dist == null)
            this.dist = new NormalDistribution(0.01,1);

    }


    /**
     * Creates and returns a copy of this object.  The precise meaning
     * of "copy" may depend on the class of the object. The general
     * intent is that, for any object {@code x}, the expression:
     * <blockquote>
     * <pre>
     * x.clone() != x</pre></blockquote>
     * will be true, and that the expression:
     * <blockquote>
     * <pre>
     * x.clone().getClass() == x.getClass()</pre></blockquote>
     * will be {@code true}, but these are not absolute requirements.
     * While it is typically the case that:
     * <blockquote>
     * <pre>
     * x.clone().equals(x)</pre></blockquote>
     * will be {@code true}, this is not an absolute requirement.
     *
     * By convention, the returned object should be obtained by calling
     * {@code super.clone}.  If a class and all of its superclasses (except
     * {@code Object}) obey this convention, it will be the case that
     * {@code x.clone().getClass() == x.getClass()}.
     *
     * By convention, the object returned by this method should be independent
     * of this object (which is being cloned).  To achieve this independence,
     * it may be necessary to modify one or more fields of the object returned
     * by {@code super.clone} before returning it.  Typically, this means
     * copying any mutable objects that comprise the internal "deep structure"
     * of the object being cloned and replacing the references to these
     * objects with references to the copies.  If a class contains only
     * primitive fields or references to immutable objects, then it is usually
     * the case that no fields in the object returned by {@code super.clone}
     * need to be modified.
     *
     * The method {@code clone} for class {@code Object} performs a
     * specific cloning operation. First, if the class of this object does
     * not implement the interface {@code Cloneable}, then a
     * {@code CloneNotSupportedException} is thrown. Note that all arrays
     * are considered to implement the interface {@code Cloneable} and that
     * the return type of the {@code clone} method of an array type {@code T[]}
     * is {@code T[]} where T is any reference or primitive type.
     * Otherwise, this method creates a new instance of the class of this
     * object and initializes all its fields with exactly the contents of
     * the corresponding fields of this object, as if by assignment; the
     * contents of the fields are not themselves cloned. Thus, this method
     * performs a "shallow copy" of this object, not a "deep copy" operation.
     *
     * The class {@code Object} does not itself implement the interface
     * {@code Cloneable}, so calling the {@code clone} method on an object
     * whose class is {@code Object} will result in throwing an
     * exception at run time.
     *
     * @return a clone of this instance.
     * @throws CloneNotSupportedException if the object's class does not
     *                                    support the {@code Cloneable} interface. Subclasses
     *                                    that overrideLayer the {@code clone} method can also
     *                                    throw this exception to indicate that an instance cannot
     *                                    be cloned.
     * @see Cloneable
     */
    @Override
    public NeuralNetConfiguration clone()  {
        return new NeuralNetConfiguration(this);
    }

    public List<String> variables() {
        return new ArrayList<>(variables);
    }

    public void addVariable(String variable) {
        if(!variables.contains(variable))
            variables.add(variable);
    }

    private static <T> T overRideFields(T configInst, Layer layer) {
        // overwrite builder with fields with layer fields
        Class<?> layerClazz = layer.getClass();
        Field[] neuralNetConfFields = Dl4jReflection.getAllFields(configInst.getClass());
        Field[] layerFields = Dl4jReflection.getAllFields(layerClazz);
        for(Field neuralNetField : neuralNetConfFields) {
            neuralNetField.setAccessible(true);
            for(Field layerField : layerFields) {
                layerField.setAccessible(true);
                if(neuralNetField.getName().equals(layerField.getName())) {
                    try {
                        Object layerFieldValue = layerField.get(layer);
                        if(layerFieldValue != null ) {
                        	if(neuralNetField.getType().isAssignableFrom(layerField.getType())){
                        		//Same class, or neuralNetField is superclass/superinterface of layer field
                        		if(!ClassUtils.isPrimitiveOrWrapper(layerField.getType()) ){
                            		neuralNetField.set(configInst, layerFieldValue);
                            	} else {
                            		//Primitive -> autoboxed by Field.get(...). Hence layerFieldValue is never null for primitive fields,
                            		// even if not explicitly set (due to default value for primitives)
                            		//Convention here is to use Double.NaN, Float.NaN, Integer.MIN_VALUE, etc. as defaults in layer configs
                            		// to signify 'not set'
                            		Class<?> primitiveClass = layerField.getType();
                            		if( primitiveClass == double.class || primitiveClass == Double.class ){
                            			if( !Double.isNaN((double)layerFieldValue) ){
                            				neuralNetField.set(configInst, layerFieldValue);
                            			}
                            		} else if( primitiveClass == float.class || primitiveClass == Float.class ){
                            			if( !Float.isNaN((float)layerFieldValue) ){
                            				neuralNetField.set(configInst, layerFieldValue);
                            			}
                            		} else if( primitiveClass == int.class || primitiveClass == Integer.class ){
                            			if( ((int)layerFieldValue) != Integer.MIN_VALUE ){
                            				neuralNetField.set(configInst, layerFieldValue);
                            			}
                            		} else if( primitiveClass == long.class || primitiveClass == Long.class ){
                            			if( ((long)layerFieldValue) != Long.MIN_VALUE ){
                            				neuralNetField.set(configInst, layerFieldValue);
                            			}
                            		} else if( primitiveClass == char.class || primitiveClass == Character.class ){
                            			if( ((char)layerFieldValue) != Character.MIN_VALUE ){
                            				neuralNetField.set(configInst, layerFieldValue);
                            			}
                            		} else {
                            			//Boolean: can only be true/false. No usable 'not set' value -> need some other workaround
                            			//Short, Byte: probably never used
                            			throw new RuntimeException("Primitive type not settable via reflection");
                            		}
                            	}
                            }
                        }
                    } catch(Exception e) {
                        throw new RuntimeException(e);
                    }
                }
            }
        }
        return configInst;
    }

        /**
         * Fluent interface for building a list of configurations
         */
    public static class ListBuilder extends MultiLayerConfiguration.Builder {
        private HashMap<Integer, Builder> layerwise;

        // Constructor
        public ListBuilder(HashMap<Integer, Builder> layerMap) {
            this.layerwise = layerMap;
        }


        public ListBuilder backward(boolean backward) {
            this.backward = backward;
            return this;
        }

        public ListBuilder hiddenLayerSizes(int...hiddenLayerSizes) {
            this.hiddenLayerSizes = hiddenLayerSizes;
            return this;
        }

        public ListBuilder layer(int ind, Layer layer) {
            if (layerwise.size() < 2) {
                throw new IllegalArgumentException("IndexOutOfBoundsError: Layer index exceeds listed size. List at least 2 layers");
            }
            if (layerwise.get(0) == null && ind != 0) {
                throw new IllegalArgumentException("LayerZeroIndexError: Layer index must start from 0");
            }
            if (layerwise.size() < ind + 1) {
                throw new IllegalArgumentException("IndexOutOfBoundsError: Layer index exceeds listed size");
            }

            Builder builderWithLayer = layerwise.get(ind).layer(layer);
            layerwise.put(ind, builderWithLayer);
            return this;
        }

        /**
         * Build the multi layer network
         * based on this neural network and
         * overr ridden parameters
         * @return the configuration to build
         */
        public MultiLayerConfiguration build() {
            List<NeuralNetConfiguration> list = new ArrayList<>();
            for(int i = 0; i < layerwise.size(); i++) {
//                if(confOverrides.get(i) != null)
//                    confOverrides.get(i).overrideLayer(i,layerwise.get(i));
                list.add(layerwise.get(i).build());
            }
            return new MultiLayerConfiguration.Builder().backward(backward).inputPreProcessors(inputPreProcessor)
                    .useDropConnect(useDropConnect).pretrain(pretrain).preProcessors(preProcessors)
                    .hiddenLayerSizes(hiddenLayerSizes)
                    .confs(list).build();
        }

    }
    /**
     * Return this configuration as json
     * @return this configuration represented as json
     */
    public String toYaml() {
        ObjectMapper mapper = mapperYaml();

        try {
            String ret =  mapper.writeValueAsString(this);
            return ret
                    .replaceAll("\"activationFunction\",","");

        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return
     */
    public static NeuralNetConfiguration fromYaml(String json) {
        ObjectMapper mapper = mapperYaml();
        try {
            NeuralNetConfiguration ret =  mapper.readValue(json, NeuralNetConfiguration.class);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Return this configuration as json
     * @return this configuration represented as json
     */
    public String toJson() {
        ObjectMapper mapper = mapper();

        try {
            String ret =  mapper.writeValueAsString(this);
            return ret
                    .replaceAll("\"activationFunction\",","");

        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return
     */
    public static NeuralNetConfiguration fromJson(String json) {
        ObjectMapper mapper = mapper();
        try {
            NeuralNetConfiguration ret =  mapper.readValue(json, NeuralNetConfiguration.class);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Object mapper for serialization of configurations
     * @return
     */
    public static ObjectMapper mapperYaml() {
        ObjectMapper ret = new ObjectMapper(new YAMLFactory());
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS,false);
        ret.enable(SerializationFeature.INDENT_OUTPUT);
        SimpleModule module = new SimpleModule();

        module.addSerializer(OutputPreProcessor.class,new PreProcessorSerializer());
        module.addDeserializer(OutputPreProcessor.class,new PreProcessorDeSerializer());

        ret.registerModule(module);
        return ret;
    }

    /**
     * Object mapper for serialization of configurations
     * @return
     */
    public static ObjectMapper mapper() {
        ObjectMapper ret = new ObjectMapper();
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS,false);
        ret.enable(SerializationFeature.INDENT_OUTPUT);
        SimpleModule module = new SimpleModule();

        module.addSerializer(OutputPreProcessor.class,new PreProcessorSerializer());
        module.addDeserializer(OutputPreProcessor.class,new PreProcessorDeSerializer());

        ret.registerModule(module);
        return ret;
    }

    public static class Builder {
        private int k = 1;
        private String customLossFunction;
        private double rmsDecay;
        @Deprecated
        private int kernel = 5;
        private double corruptionLevel = 3e-1f;
        private double sparsity = 0f;
        @Deprecated
        private boolean useAdaGrad = true;
        private double lr = 1e-1f;
        private double momentum = 0.5f;
        private double l2 = 0f;
        private boolean useRegularization = false;
        private Map<Integer, Double> momentumAfter;
        private int resetAdaGradIterations = -1;
        private double dropOut = 0;
        private boolean applySparsity = false;
        private WeightInit weightInit = WeightInit.VI;
        private OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.CONJUGATE_GRADIENT;
        private boolean constrainGradientToUnitNorm = false;
        @Deprecated
        private DefaultRandom rng = null;
        private long seed = System.currentTimeMillis();
        @Deprecated
        private Distribution dist  = new NormalDistribution(1e-3,1);
        private LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY;
        private int nIn;
        private int nOut;
        private String activationFunction = "sigmoid";
        private RBM.VisibleUnit visibleUnit = RBM.VisibleUnit.BINARY;
        private RBM.HiddenUnit hiddenUnit = RBM.HiddenUnit.BINARY;
        private int numIterations = 5;
        private int[] weightShape;
        private int[] filterSize = {2,2};
        private int filterDepth = 5;
        @Deprecated
        private int[] featureMapSize = {2,2};
        //subsampling layers
        private int[] stride = {2,2};
        private StepFunction stepFunction = new NegativeDefaultStepFunction();
        private Layer layer;
        private int batchSize = 100;
        @Deprecated
        private int numLineSearchIterations = 5;
        private int maxNumLineSearchIterations = 5;
        private boolean minimize = false;
        private Convolution.Type convolutionType = Convolution.Type.VALID;
        private SubsamplingLayer.poolingType poolingType = SubsamplingLayer.poolingType.MAX;
        private double l1 = 0.0;
        private boolean useDropConnect = false;
        private double rho;
        private Updater updater = Updater.ADAGRAD;
        private int channels = 1;
        private boolean miniBatch = false;

        /**
         * Number of channels for a conv net
         *
         * @param channels
         * @return
         */
        public Builder channels(int channels) {
            this.channels = channels;
            return this;
        }


        /**
         * The updater to use
         * @param updater
         * @return
         */
        public Builder updater(Updater updater) {
            this.updater = updater;
            return this;
        }

        /**
         * Ada delta coefficient
         * @param rho
         * @return
         */
        public Builder rho(double rho) {
            this.rho = rho;
            return this;
        }


        public Builder miniBatch(boolean miniBatch) {
            this.miniBatch = miniBatch;
            return this;
        }


        /**
         * Use drop connect: multiply the coefficients
         * by a binomial sampling wrt the dropout probability
         * @param useDropConnect whether to use drop connect or not
         * @return the
         */
        public Builder useDropConnect(boolean useDropConnect) {
            this.useDropConnect = useDropConnect;
            return this;
        }

        public Builder l1(double l1) {
            this.l1 = l1;
            return this;
        }


        public Builder rmsDecay(double rmsDecay) {
            this.rmsDecay = rmsDecay;
            return this;
        }

        public Builder customLossFunction(String customLossFunction) {
            this.customLossFunction = customLossFunction;
            return this;
        }

        public Builder convolutionType(Convolution.Type convolutionType) {
            this.convolutionType = convolutionType;
            return this;
        }

        public Builder poolingType(SubsamplingLayer.poolingType poolingType) {
            this.poolingType = poolingType;
            return this;
        }

        public Builder minimize(boolean minimize) {
            this.minimize = minimize;
            return this;
        }

        @Deprecated
        public Builder numLineSearchIterations(int numLineSearchIterations) {
            this.numLineSearchIterations = numLineSearchIterations;
            return this;
        }

        public Builder maxNumLineSearchIterations(int maxNumLineSearchIterations) {
            this.maxNumLineSearchIterations = maxNumLineSearchIterations;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder kernel(int kernel) {
            this.kernel = kernel;
            return this;
        }

        public Builder layer(Layer layer) {
            this.layer = layer;
            return this;
        }

        public Builder stepFunction(StepFunction stepFunction) {
            this.stepFunction = stepFunction;
            return this;
        }

        public ListBuilder list(int size) {
            if(size < 2)
                throw new IllegalArgumentException("Number of layers must be > 1");

            HashMap<Integer, Builder> layerMap = new HashMap<>();
            for(int i = 0; i < size; i++)
                layerMap.put(i, clone());
            return new ListBuilder(layerMap);
        }

        public Builder clone() {
            Builder b = new Builder();
            Field[] fields = Builder.class.getDeclaredFields();
            for(Field f : fields) {
                try {
                    f.setAccessible(true);
                    f.set(b,f.get(this));
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
            }

            return b;
        }

        public Builder featureMapSize(int...featureMapSize) {
            this.featureMapSize = featureMapSize;
            return this;
        }


        public Builder stride(int[] stride) {
            if(stride.length != 2)
                throw new IllegalArgumentException("Invalid stride  must be length 2");
            this.stride = stride;
            return this;
        }

        public Builder filterSize(int...filterSize) {
            if(filterSize.length != 2)
                throw new IllegalArgumentException("Invalid filter size must be length 2");
            this.filterSize = filterSize;
            return this;
        }

        public Builder filterDepth(int filterDepth) {
            this.filterDepth = filterDepth;
            return this;
        }

        public Builder weightShape(int[] weightShape) {
            this.weightShape = weightShape;
            return this;
        }

        public Builder iterations(int numIterations) {
            this.numIterations = numIterations;
            return this;
        }

        public Builder dist(Distribution dist) {
            this.dist = dist;
            return this;
        }

        public Builder sparsity(double sparsity) {
            this.sparsity = sparsity;
            return this;
        }

        @Deprecated
        public Builder useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }

        public Builder learningRate(double lr) {
            this.lr = lr;
            return this;
        }

        public Builder momentum(double momentum) {
            this.momentum = momentum;
            return this;
        }

        public Builder k(int k) {
            this.k = k;
            return this;
        }

        public Builder corruptionLevel(double corruptionLevel) {
            this.corruptionLevel = corruptionLevel;
            return this;
        }

        public Builder momentumAfter(Map<Integer, Double> momentumAfter) {
            this.momentumAfter = momentumAfter;
            return this;
        }

        @Deprecated
        public Builder adagradResetIterations(int resetAdaGradIterations) {
            this.resetAdaGradIterations = resetAdaGradIterations;
            return this;
        }

        public Builder dropOut(double dropOut) {
            this.dropOut = dropOut;
            return this;
        }

        public Builder applySparsity(boolean applySparsity) {
            this.applySparsity = applySparsity;
            return this;
        }

        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }

        @Deprecated
        public Builder rng(DefaultRandom rng) {
            this.rng = rng;
            return this;
        }

        public Builder seed(int seed) {
            this.seed = (long) seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }


        public Builder l2(double l2) {
            this.l2 = l2;
            return this;
        }

        public Builder regularization(boolean useRegularization) {
            this.useRegularization = useRegularization;
            return this;
        }

        public Builder resetAdaGradIterations(int resetAdaGradIterations) {
            this.resetAdaGradIterations = resetAdaGradIterations;
            return this;
        }

        public Builder optimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
            return this;
        }

        public Builder lossFunction(LossFunctions.LossFunction lossFunction) {
            this.lossFunction = lossFunction;
            return this;
        }

        public Builder constrainGradientToUnitNorm(boolean constrainGradientToUnitNorm) {
            this.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
            return this;
        }

        public Builder nIn(int nIn) {
            this.nIn = nIn;
            return this;
        }

        public Builder nOut(int nOut) {
            this.nOut = nOut;
            return this;
        }

        public Builder activationFunction(String activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public Builder visibleUnit(RBM.VisibleUnit visibleUnit) {
            this.visibleUnit = visibleUnit;
            return this;
        }

        public Builder hiddenUnit(RBM.HiddenUnit hiddenUnit) {
            this.hiddenUnit = hiddenUnit;
            return this;
        }

        /**
         * Return a configuration based on this builder
         *
         * @return
         */
        public NeuralNetConfiguration build() {
            if (layer == null)
                throw new IllegalStateException("No layer defined.");

            NeuralNetConfiguration ret = new NeuralNetConfiguration( sparsity,  useAdaGrad,  lr,  k,
                    corruptionLevel,  numIterations,  momentum,  l2,  useRegularization, momentumAfter,
                    resetAdaGradIterations,  dropOut,  applySparsity,  weightInit,  optimizationAlgo, lossFunction,
                    constrainGradientToUnitNorm,  rng, seed,
                    dist,  nIn,  nOut,  activationFunction, visibleUnit,hiddenUnit,weightShape,filterSize, filterDepth, stride,featureMapSize,kernel
                    ,batchSize,numLineSearchIterations,maxNumLineSearchIterations,minimize,layer,convolutionType,poolingType,
                    l1,customLossFunction);
            ret.useAdaGrad = this.useAdaGrad;
            ret.rmsDecay = rmsDecay;
            ret.stepFunction = stepFunction;
            ret.useDropConnect = useDropConnect;
            ret.miniBatch = miniBatch;
            ret.rho = rho;
            ret.updater = updater;

            //override the properties from the layer
            ret = overRideFields(ret, layer);
            return ret;
        }
    }
}
