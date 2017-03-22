/*-
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

import com.google.common.collect.Sets;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.lang3.ClassUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.util.reflections.DL4JSubTypesScanner;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.*;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.databind.introspect.AnnotatedClass;
import org.nd4j.shade.jackson.databind.jsontype.NamedType;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import org.reflections.ReflectionUtils;
import org.reflections.Reflections;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;
import org.reflections.util.FilterBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Serializable;
import java.lang.reflect.Modifier;
import java.net.URL;
import java.util.*;


/**
 * A Serializable configuration
 * for neural nets that covers per layer parameters
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class NeuralNetConfiguration implements Serializable, Cloneable {

    private static final Logger log = LoggerFactory.getLogger(NeuralNetConfiguration.class);

    /**
     * System property for custom layers, preprocessors, graph vertices etc. Enabled by default.
     * Run JVM with "-Dorg.deeplearning4j.config.custom.enabled=false" to disable classpath scanning for
     * Overriding the default (i.e., disabling) this is only useful if (a) no custom layers/preprocessors etc will be
     * used, and (b) minimizing startup/initialization time for new JVMs is very important.
     * Results are cached, so there is no cost to custom layers after the first network has been constructed.
     */
    public static final String CUSTOM_FUNCTIONALITY = "org.deeplearning4j.config.custom.enabled";

    protected Layer layer;
    @Deprecated
    protected double leakyreluAlpha;
    //batch size: primarily used for conv nets. Will be reinforced if set.
    protected boolean miniBatch = true;
    protected int numIterations;
    //number of line search iterations
    protected int maxNumLineSearchIterations;
    protected long seed;
    protected OptimizationAlgorithm optimizationAlgo;
    //gradient keys used for ensuring order when getting and setting the gradient
    protected List<String> variables = new ArrayList<>();
    //whether to constrain the gradient to unit norm or not
    //adadelta - weight for how much to consider previous history
    protected StepFunction stepFunction;
    protected boolean useRegularization = false;
    protected boolean useDropConnect = false;
    //minimize or maximize objective
    protected boolean minimize = true;
    // Graves LSTM & RNN
    protected Map<String, Double> learningRateByParam = new HashMap<>();
    protected Map<String, Double> l1ByParam = new HashMap<>();
    protected Map<String, Double> l2ByParam = new HashMap<>();
    protected LearningRatePolicy learningRatePolicy = LearningRatePolicy.None;
    protected double lrPolicyDecayRate;
    protected double lrPolicySteps;
    protected double lrPolicyPower;
    protected boolean pretrain;

    //Counter for the number of parameter updates so far for this layer.
    //Note that this is only used for pretrain layers (RBM, VAE) - MultiLayerConfiguration and ComputationGraphConfiguration
    //contain counters for standard backprop training.
    // This is important for learning rate schedules, for example, and is stored here to ensure it is persisted
    // for Spark and model serialization
    protected int iterationCount = 0;


    private static ObjectMapper mapper = initMapper();
    private static final ObjectMapper mapperYaml = initMapperYaml();
    private static Set<Class<?>> subtypesClassCache = null;


    /**
     * Creates and returns a deep copy of the configuration.
     */
    @Override
    public NeuralNetConfiguration clone() {
        try {
            NeuralNetConfiguration clone = (NeuralNetConfiguration) super.clone();
            if (clone.layer != null)
                clone.layer = clone.layer.clone();
            if (clone.stepFunction != null)
                clone.stepFunction = clone.stepFunction.clone();
            if (clone.variables != null)
                clone.variables = new ArrayList<>(clone.variables);
            if (clone.learningRateByParam != null)
                clone.learningRateByParam = new HashMap<>(clone.learningRateByParam);
            if (clone.l1ByParam != null)
                clone.l1ByParam = new HashMap<>(clone.l1ByParam);
            if (clone.l2ByParam != null)
                clone.l2ByParam = new HashMap<>(clone.l2ByParam);
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public List<String> variables() {
        return new ArrayList<>(variables);
    }

    public List<String> variables(boolean copy) {
        if (copy)
            return variables();
        return variables;
    }

    public void addVariable(String variable) {
        if (!variables.contains(variable)) {
            variables.add(variable);
            setLayerParamLR(variable);
        }
    }

    public void clearVariables() {
        variables.clear();
    }

    public void setLayerParamLR(String variable) {
        double lr = layer.getLearningRateByParam(variable);
        double l1 = layer.getL1ByParam(variable);
        if (Double.isNaN(l1))
            l1 = 0.0; //Not set
        double l2 = layer.getL2ByParam(variable);
        if (Double.isNaN(l2))
            l2 = 0.0; //Not set
        learningRateByParam.put(variable, lr);
        l1ByParam.put(variable, l1);
        l2ByParam.put(variable, l2);
    }

    public double getLearningRateByParam(String variable) {
        return learningRateByParam.get(variable);
    }

    public void setLearningRateByParam(String variable, double rate) {
        learningRateByParam.put(variable, rate);
    }

    public double getL1ByParam(String variable) {
        return l1ByParam.get(variable);
    }

    public double getL2ByParam(String variable) {
        return l2ByParam.get(variable);
    }

    /**
     * Fluent interface for building a list of configurations
     */
    public static class ListBuilder extends MultiLayerConfiguration.Builder {
        private Map<Integer, Builder> layerwise;
        private Builder globalConfig;

        // Constructor
        public ListBuilder(Builder globalConfig, Map<Integer, Builder> layerMap) {
            this.globalConfig = globalConfig;
            this.layerwise = layerMap;
        }

        public ListBuilder(Builder globalConfig) {
            this(globalConfig, new HashMap<Integer, Builder>());
        }

        public ListBuilder backprop(boolean backprop) {
            this.backprop = backprop;

            return this;
        }

        public ListBuilder pretrain(boolean pretrain) {
            this.pretrain = pretrain;
            return this;
        }

        public ListBuilder layer(int ind, Layer layer) {
            if (layerwise.containsKey(ind)) {
                layerwise.get(ind).layer(layer);
            } else {
                layerwise.put(ind, globalConfig.clone().layer(layer));
            }
            return this;
        }

        public Map<Integer, Builder> getLayerwise() {
            return layerwise;
        }

        /**
         * Build the multi layer network
         * based on this neural network and
         * overr ridden parameters
         * @return the configuration to build
         */
        public MultiLayerConfiguration build() {
            List<NeuralNetConfiguration> list = new ArrayList<>();
            if (layerwise.isEmpty())
                throw new IllegalStateException("Invalid configuration: no layers defined");
            for (int i = 0; i < layerwise.size(); i++) {
                if (layerwise.get(i) == null) {
                    throw new IllegalStateException("Invalid configuration: layer number " + i
                                    + " not specified. Expect layer " + "numbers to be 0 to " + (layerwise.size() - 1)
                                    + " inclusive (number of layers defined: " + layerwise.size() + ")");
                }
                if (layerwise.get(i).getLayer() == null)
                    throw new IllegalStateException("Cannot construct network: Layer config for" + "layer with index "
                                    + i + " is not defined)");

                //Layer names: set to default, if not set
                if (layerwise.get(i).getLayer().getLayerName() == null) {
                    layerwise.get(i).getLayer().setLayerName("layer" + i);
                }

                list.add(layerwise.get(i).build());
            }
            return new MultiLayerConfiguration.Builder().backprop(backprop).inputPreProcessors(inputPreProcessors)
                            .pretrain(pretrain).backpropType(backpropType).tBPTTForwardLength(tbpttFwdLength)
                            .tBPTTBackwardLength(tbpttBackLength).cnnInputSize(this.cnnInputSize)
                            .setInputType(this.inputType).confs(list).build();
        }

    }

    /**
     * Return this configuration as json
     * @return this configuration represented as json
     */
    public String toYaml() {
        ObjectMapper mapper = mapperYaml();

        try {
            String ret = mapper.writeValueAsString(this);
            return ret;

        } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
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
            NeuralNetConfiguration ret = mapper.readValue(json, NeuralNetConfiguration.class);
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
            String ret = mapper.writeValueAsString(this);
            return ret;

        } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
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
            NeuralNetConfiguration ret = mapper.readValue(json, NeuralNetConfiguration.class);
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
        return mapperYaml;
    }

    private static ObjectMapper initMapperYaml() {
        ObjectMapper ret = new ObjectMapper(new YAMLFactory());
        configureMapper(ret);
        return ret;
    }

    /**
     * Object mapper for serialization of configurations
     * @return
     */
    public static ObjectMapper mapper() {
        return mapper;
    }

    /**Reinitialize and return the Jackson/json ObjectMapper with additional named types.
     * This can be used to add additional subtypes at runtime (i.e., for JSON mapping with
     * types defined outside of the main DL4J codebase)
     */
    public static ObjectMapper reinitMapperWithSubtypes(Collection<NamedType> additionalTypes) {
        mapper.registerSubtypes(additionalTypes.toArray(new NamedType[additionalTypes.size()]));
        //Recreate the mapper (via copy), as mapper won't use registered subtypes after first use
        mapper = mapper.copy();
        return mapper;
    }

    private static ObjectMapper initMapper() {
        ObjectMapper ret = new ObjectMapper();
        configureMapper(ret);
        return ret;
    }

    private static void configureMapper(ObjectMapper ret) {
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        ret.enable(SerializationFeature.INDENT_OUTPUT);

        registerSubtypes(ret);
    }

    private static synchronized void registerSubtypes(ObjectMapper mapper) {
        //Register concrete subtypes for JSON serialization

        List<Class<?>> classes = Arrays.<Class<?>>asList(InputPreProcessor.class, ILossFunction.class,
                        IActivation.class, Layer.class, GraphVertex.class, ReconstructionDistribution.class);
        List<String> classNames = new ArrayList<>(6);
        for (Class<?> c : classes)
            classNames.add(c.getName());

        // First: scan the classpath and find all instances of the 'baseClasses' classes
        if (subtypesClassCache == null) {

            //Check system property:
            String prop = System.getProperty(CUSTOM_FUNCTIONALITY);
            if (prop != null && !Boolean.parseBoolean(prop)) {

                subtypesClassCache = Collections.emptySet();
            } else {

                List<Class<?>> interfaces = Arrays.<Class<?>>asList(InputPreProcessor.class, ILossFunction.class,
                                IActivation.class, ReconstructionDistribution.class);
                List<Class<?>> classesList = Arrays.<Class<?>>asList(Layer.class, GraphVertex.class);

                Collection<URL> urls = ClasspathHelper.forClassLoader();
                List<URL> scanUrls = new ArrayList<>();
                for (URL u : urls) {
                    String path = u.getPath();
                    if (!path.matches(".*/jre/lib/.*jar")) { //Skip JRE/JDK JARs
                        scanUrls.add(u);
                    }
                }

                Reflections reflections = new Reflections(new ConfigurationBuilder().filterInputsBy(new FilterBuilder()
                                .exclude("^(?!.*\\.class$).*$") //Consider only .class files (to avoid debug messages etc. on .dlls, etc
                                //Exclude the following: the assumption here is that no custom functionality will ever be present
                                // under these package name prefixes. These are all common dependencies for DL4J
                                .exclude("^org.nd4j.*").exclude("^org.datavec.*").exclude("^org.bytedeco.*") //JavaCPP
                                .exclude("^com.fasterxml.*")//Jackson
                                .exclude("^org.apache.*") //Apache commons, Spark, log4j etc
                                .exclude("^org.projectlombok.*").exclude("^com.twelvemonkeys.*").exclude("^org.joda.*")
                                .exclude("^org.slf4j.*").exclude("^com.google.*").exclude("^org.reflections.*")
                                .exclude("^ch.qos.*") //Logback
                ).addUrls(scanUrls).setScanners(new DL4JSubTypesScanner(interfaces, classesList)));
                org.reflections.Store store = reflections.getStore();

                Iterable<String> subtypesByName = store.getAll(DL4JSubTypesScanner.class.getSimpleName(), classNames);

                Set<? extends Class<?>> subtypeClasses = Sets.newHashSet(ReflectionUtils.forNames(subtypesByName));
                subtypesClassCache = new HashSet<>();
                for (Class<?> c : subtypeClasses) {
                    if (Modifier.isAbstract(c.getModifiers()) || Modifier.isInterface(c.getModifiers())) {
                        //log.info("Skipping abstract/interface: {}",c);
                        continue;
                    }
                    subtypesClassCache.add(c);
                }
            }
        }

        //Second: get all currently registered subtypes for this mapper
        Set<Class<?>> registeredSubtypes = new HashSet<>();
        for (Class<?> c : classes) {
            AnnotatedClass ac = AnnotatedClass.construct(c, mapper.getSerializationConfig().getAnnotationIntrospector(),
                            null);
            Collection<NamedType> types =
                            mapper.getSubtypeResolver().collectAndResolveSubtypes(ac, mapper.getSerializationConfig(),
                                            mapper.getSerializationConfig().getAnnotationIntrospector());
            for (NamedType nt : types) {
                registeredSubtypes.add(nt.getType());
            }
        }

        //Third: register all _concrete_ subtypes that are not already registered
        List<NamedType> toRegister = new ArrayList<>();
        for (Class<?> c : subtypesClassCache) {
            //Check if it's concrete or abstract...
            if (Modifier.isAbstract(c.getModifiers()) || Modifier.isInterface(c.getModifiers())) {
                //log.info("Skipping abstract/interface: {}",c);
                continue;
            }

            if (!registeredSubtypes.contains(c)) {
                String name;
                if (ClassUtils.isInnerClass(c)) {
                    Class<?> c2 = c.getDeclaringClass();
                    name = c2.getSimpleName() + "$" + c.getSimpleName();
                } else {
                    name = c.getSimpleName();
                }
                toRegister.add(new NamedType(c, name));
                if (log.isDebugEnabled()) {
                    for (Class<?> baseClass : classes) {
                        if (baseClass.isAssignableFrom(c)) {
                            log.debug("Registering class for JSON serialization: {} as subtype of {}", c.getName(),
                                            baseClass.getName());
                            break;
                        }
                    }
                }
            }
        }

        mapper.registerSubtypes(toRegister.toArray(new NamedType[toRegister.size()]));
    }

    @Data
    public static class Builder implements Cloneable {
        protected IActivation activationFn = new ActivationSigmoid();
        protected WeightInit weightInit = WeightInit.XAVIER;
        protected double biasInit = 0.0;
        protected Distribution dist = null;
        protected double learningRate = 1e-1;
        protected double biasLearningRate = Double.NaN;
        protected Map<Integer, Double> learningRateSchedule = null;
        protected double lrScoreBasedDecay;
        protected double l1 = Double.NaN;
        protected double l2 = Double.NaN;
        protected double l1Bias = Double.NaN;
        protected double l2Bias = Double.NaN;
        protected double dropOut = 0;
        protected Updater updater = Updater.SGD;
        protected double momentum = Double.NaN;
        protected Map<Integer, Double> momentumSchedule = null;
        protected double epsilon = Double.NaN;
        protected double rho = Double.NaN;
        protected double rmsDecay = Double.NaN;
        protected double adamMeanDecay = Double.NaN;
        protected double adamVarDecay = Double.NaN;
        protected Layer layer;
        @Deprecated
        protected double leakyreluAlpha = 0.01;
        protected boolean miniBatch = true;
        protected int numIterations = 1;
        protected int maxNumLineSearchIterations = 5;
        protected long seed = System.currentTimeMillis();
        protected boolean useRegularization = false;
        protected OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
        protected StepFunction stepFunction = null;
        protected boolean useDropConnect = false;
        protected boolean minimize = true;
        protected GradientNormalization gradientNormalization = GradientNormalization.None;
        protected double gradientNormalizationThreshold = 1.0;
        protected LearningRatePolicy learningRatePolicy = LearningRatePolicy.None;
        protected double lrPolicyDecayRate = Double.NaN;
        protected double lrPolicySteps = Double.NaN;
        protected double lrPolicyPower = Double.NaN;
        protected boolean pretrain = false;

        protected ConvolutionMode convolutionMode = ConvolutionMode.Truncate;

        public Builder() {
            //
        }

        public Builder(NeuralNetConfiguration newConf) {
            if (newConf != null) {
                minimize = newConf.minimize;
                maxNumLineSearchIterations = newConf.maxNumLineSearchIterations;
                layer = newConf.layer;
                numIterations = newConf.numIterations;
                useRegularization = newConf.useRegularization;
                optimizationAlgo = newConf.optimizationAlgo;
                seed = newConf.seed;
                stepFunction = newConf.stepFunction;
                useDropConnect = newConf.useDropConnect;
                miniBatch = newConf.miniBatch;
                learningRatePolicy = newConf.learningRatePolicy;
                lrPolicyDecayRate = newConf.lrPolicyDecayRate;
                lrPolicySteps = newConf.lrPolicySteps;
                lrPolicyPower = newConf.lrPolicyPower;
                pretrain = newConf.pretrain;
            }
        }

        /** Process input as minibatch vs full dataset.
         * Default set to true. */
        public Builder miniBatch(boolean miniBatch) {
            this.miniBatch = miniBatch;
            return this;
        }

        /**
         * Use drop connect: multiply the weight by a binomial sampling wrt the dropout probability.
         * Dropconnect probability is set using {@link #dropOut(double)}; this is the probability of retaining a weight
         * @param useDropConnect whether to use drop connect or not
         * @return the
         */
        public Builder useDropConnect(boolean useDropConnect) {
            this.useDropConnect = useDropConnect;
            return this;
        }

        /** Objective function to minimize or maximize cost function
         * Default set to minimize true. */
        public Builder minimize(boolean minimize) {
            this.minimize = minimize;
            return this;
        }

        /** Maximum number of line search iterations.
         * Only applies for line search optimizers: Line Search SGD, Conjugate Gradient, LBFGS
         * is NOT applicable for standard SGD
         * @param maxNumLineSearchIterations > 0
         * @return
         */
        public Builder maxNumLineSearchIterations(int maxNumLineSearchIterations) {
            this.maxNumLineSearchIterations = maxNumLineSearchIterations;
            return this;
        }



        /** Layer class. */
        public Builder layer(Layer layer) {
            this.layer = layer;
            return this;
        }

        /** Step function to apply for back track line search.
         * Only applies for line search optimizers: Line Search SGD, Conjugate Gradient, LBFGS
         * Options: DefaultStepFunction (default), NegativeDefaultStepFunction
         * GradientStepFunction (for SGD), NegativeGradientStepFunction */
        public Builder stepFunction(StepFunction stepFunction) {
            this.stepFunction = stepFunction;
            return this;
        }

        /**Create a ListBuilder (for creating a MultiLayerConfiguration)<br>
         * Usage:<br>
         * <pre>
         * {@code .list()
         * .layer(0,new DenseLayer.Builder()...build())
         * ...
         * .layer(n,new OutputLayer.Builder()...build())
         * }
         * </pre>
         * */
        public ListBuilder list() {
            return new ListBuilder(this);
        }

        /**Create a ListBuilder (for creating a MultiLayerConfiguration) with the specified layers<br>
         * Usage:<br>
         * <pre>
         * {@code .list(
         *      new DenseLayer.Builder()...build(),
         *      ...,
         *      new OutputLayer.Builder()...build())
         * }
         * </pre>
         * @param layers The layer configurations for the network
         */
        public ListBuilder list(Layer... layers) {
            if (layers == null || layers.length == 0)
                throw new IllegalArgumentException("Cannot create network with no layers");
            Map<Integer, Builder> layerMap = new HashMap<>();
            for (int i = 0; i < layers.length; i++) {
                Builder b = this.clone();
                b.layer(layers[i]);
                layerMap.put(i, b);
            }
            return new ListBuilder(this, layerMap);

        }

        /**
         * Create a GraphBuilder (for creating a ComputationGraphConfiguration).
         */
        public ComputationGraphConfiguration.GraphBuilder graphBuilder() {
            return new ComputationGraphConfiguration.GraphBuilder(this);
        }

        /** Number of optimization iterations. */
        public Builder iterations(int numIterations) {
            this.numIterations = numIterations;
            return this;
        }

        /** Random number generator seed. Used for reproducability between runs */
        public Builder seed(int seed) {
            this.seed = (long) seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }

        /** Random number generator seed. Used for reproducability between runs */
        public Builder seed(long seed) {
            this.seed = seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }

        /**
         * Optimization algorithm to use. Most common: OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
         *
         * @param optimizationAlgo    Optimization algorithm to use when training
         */
        public Builder optimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
            return this;
        }

        /** Whether to use regularization (l1, l2, dropout, etc */
        public Builder regularization(boolean useRegularization) {
            this.useRegularization = useRegularization;
            return this;
        }

        @Override
        public Builder clone() {
            try {
                Builder clone = (Builder) super.clone();
                if (clone.layer != null)
                    clone.layer = clone.layer.clone();
                if (clone.stepFunction != null)
                    clone.stepFunction = clone.stepFunction.clone();

                return clone;

            } catch (CloneNotSupportedException e) {
                throw new RuntimeException(e);
            }
        }

        /**Activation function / neuron non-linearity
         * Typical values include:<br>
         * "relu" (rectified linear), "tanh", "sigmoid", "softmax",
         * "hardtanh", "leakyrelu", "maxout", "softsign", "softplus"
         * @deprecated Use {@link #activation(Activation)} or
         * {@link @activation(IActivation)}
         */
        @Deprecated
        public Builder activation(String activationFunction) {
            return activation(Activation.fromString(activationFunction).getActivationFunction());
        }

        /**Activation function / neuron non-linearity
         * @see #activation(Activation)
         */
        public Builder activation(IActivation activationFunction) {
            this.activationFn = activationFunction;
            return this;
        }

        /**Activation function / neuron non-linearity
         */
        public Builder activation(Activation activation) {
            return activation(activation.getActivationFunction());
        }

        /**
         * @deprecated Use {@link #activation(IActivation)} with leaky relu, setting alpha value directly in constructor.
         */
        @Deprecated
        public Builder leakyreluAlpha(double leakyreluAlpha) {
            this.leakyreluAlpha = leakyreluAlpha;
            return this;
        }

        /** Weight initialization scheme.
         * @see org.deeplearning4j.nn.weights.WeightInit
         */
        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }

        /**
         * Constant for bias initialization. Default: 0.0
         *
         * @param biasInit    Constant for bias initialization
         */
        public Builder biasInit(double biasInit) {
            this.biasInit = biasInit;
            return this;
        }

        /** Distribution to sample initial weights from. Used in conjunction with
         * .weightInit(WeightInit.DISTRIBUTION).
         */
        public Builder dist(Distribution dist) {
            this.dist = dist;
            return this;
        }

        /** Learning rate. Defaults to 1e-1*/
        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        /** Bias learning rate. Set this to apply a different learning rate to the bias*/
        public Builder biasLearningRate(double biasLearningRate) {
            this.biasLearningRate = biasLearningRate;
            return this;
        }

        /** Learning rate schedule. Map of the iteration to the learning rate to apply at that iteration. */
        public Builder learningRateSchedule(Map<Integer, Double> learningRateSchedule) {
            this.learningRateSchedule = learningRateSchedule;
            return this;
        }

        /** Rate to decrease learningRate by when the score stops improving.
         * Learning rate is multiplied by this rate so ideally keep between 0 and 1. */
        public Builder learningRateScoreBasedDecayRate(double lrScoreBasedDecay) {
            this.lrScoreBasedDecay = lrScoreBasedDecay;
            return this;
        }

        /** L1 regularization coefficient for the weights.
         *  Use with .regularization(true)
         */
        public Builder l1(double l1) {
            this.l1 = l1;
            return this;
        }

        /** L2 regularization coefficient for the weights.
         * Use with .regularization(true)
         */
        public Builder l2(double l2) {
            this.l2 = l2;
            return this;
        }

        /** L1 regularization coefficient for the bias.
         *  Use with .regularization(true)
         */
        public Builder l1Bias(double l1Bias) {
            this.l1Bias = l1Bias;
            return this;
        }

        /** L2 regularization coefficient for the bias.
         * Use with .regularization(true)
         */
        public Builder l2Bias(double l2Bias) {
            this.l2Bias = l2Bias;
            return this;
        }

        /**
         * Dropout probability. This is the probability of <it>retaining</it> an activation. So dropOut(x) will keep an
         * activation with probability x, and set to 0 with probability 1-x.<br>
         * dropOut(0.0) is disabled (default).
         *
         * @param dropOut    Dropout probability (probability of retaining an activation)
         */
        public Builder dropOut(double dropOut) {
            this.dropOut = dropOut;
            return this;
        }

        /** Momentum rate
         *  Used only when Updater is set to {@link Updater#NESTEROVS}
         */
        public Builder momentum(double momentum) {
            this.momentum = momentum;
            return this;
        }

        /** Momentum schedule. Map of the iteration to the momentum rate to apply at that iteration
         *  Used only when Updater is set to {@link Updater#NESTEROVS}
         */
        public Builder momentumAfter(Map<Integer, Double> momentumAfter) {
            this.momentumSchedule = momentumAfter;
            return this;
        }

        /** Gradient updater. For example, Updater.SGD for standard stochastic gradient descent,
         * Updater.NESTEROV for Nesterov momentum, Updater.RSMPROP for RMSProp, etc.
         * @see Updater
         */
        public Builder updater(Updater updater) {
            this.updater = updater;
            return this;
        }

        /**
         * Ada delta coefficient
         * @param rho
         */
        public Builder rho(double rho) {
            this.rho = rho;
            return this;
        }


        /**
         * Epsilon value for updaters: Adam, RMSProp, Adagrad, Adadelta
         * Default values: {@link Adam#DEFAULT_ADAM_EPSILON}, {@link RmsProp#DEFAULT_RMSPROP_EPSILON}, {@link AdaGrad#DEFAULT_ADAGRAD_EPSILON},
         * {@link AdaDelta#DEFAULT_ADADELTA_EPSILON}
         *
         * @param epsilon    Epsilon value to use for adagrad or
         */
        public Builder epsilon(double epsilon) {
            this.epsilon = epsilon;
            return this;
        }

        /** Decay rate for RMSProp. Only applies if using .updater(Updater.RMSPROP)
         */
        public Builder rmsDecay(double rmsDecay) {
            this.rmsDecay = rmsDecay;
            return this;
        }

        /** Mean decay rate for Adam updater. Only applies if using .updater(Updater.ADAM) */
        public Builder adamMeanDecay(double adamMeanDecay) {
            this.adamMeanDecay = adamMeanDecay;
            return this;
        }

        /** Variance decay rate for Adam updater. Only applies if using .updater(Updater.ADAM) */
        public Builder adamVarDecay(double adamVarDecay) {
            this.adamVarDecay = adamVarDecay;
            return this;
        }

        /** Gradient normalization strategy. Used to specify gradient renormalization, gradient clipping etc.
         * @param gradientNormalization Type of normalization to use. Defaults to None.
         * @see GradientNormalization
         */
        public Builder gradientNormalization(GradientNormalization gradientNormalization) {
            this.gradientNormalization = gradientNormalization;
            return this;
        }

        /** Threshold for gradient normalization, only used for GradientNormalization.ClipL2PerLayer,
         * GradientNormalization.ClipL2PerParamType, and GradientNormalization.ClipElementWiseAbsoluteValue<br>
         * Not used otherwise.<br>
         * L2 threshold for first two types of clipping, or absolute value threshold for last type of clipping.
         */
        public Builder gradientNormalizationThreshold(double threshold) {
            this.gradientNormalizationThreshold = threshold;
            return this;
        }

        /** Learning rate decay policy. Used to adapt learning rate based on policy.
         * @param policy Type of policy to use. Defaults to None.
         */
        public Builder learningRateDecayPolicy(LearningRatePolicy policy) {
            this.learningRatePolicy = policy;
            return this;
        }

        /** Set the decay rate for the learning rate decay policy.
         * @param lrPolicyDecayRate rate.
         */
        public Builder lrPolicyDecayRate(double lrPolicyDecayRate) {
            this.lrPolicyDecayRate = lrPolicyDecayRate;
            return this;
        }

        /** Set the number of steps used for learning decay rate steps policy.
         * @param lrPolicySteps number of steps
         */
        public Builder lrPolicySteps(double lrPolicySteps) {
            this.lrPolicySteps = lrPolicySteps;
            return this;
        }

        /** Set the power used for learning rate inverse policy.
         * @param lrPolicyPower power
         */
        public Builder lrPolicyPower(double lrPolicyPower) {
            this.lrPolicyPower = lrPolicyPower;
            return this;
        }

        public Builder convolutionMode(ConvolutionMode convolutionMode) {
            this.convolutionMode = convolutionMode;
            return this;
        }

        private void learningRateValidation(String layerName) {
            if (learningRatePolicy != LearningRatePolicy.None && Double.isNaN(lrPolicyDecayRate)) {
                //LR policy, if used, should have a decay rate. 2 exceptions: Map for schedule, and Poly + power param
                if (!(learningRatePolicy == LearningRatePolicy.Schedule && learningRateSchedule != null)
                                && !(learningRatePolicy == LearningRatePolicy.Poly && !Double.isNaN(lrPolicyPower)))
                    throw new IllegalStateException("Layer \"" + layerName
                                    + "\" learning rate policy decay rate (lrPolicyDecayRate) must be set to use learningRatePolicy.");
            }
            switch (learningRatePolicy) {
                case Inverse:
                case Poly:
                    if (Double.isNaN(lrPolicyPower))
                        throw new IllegalStateException("Layer \"" + layerName
                                        + "\" learning rate policy power (lrPolicyPower) must be set to use "
                                        + learningRatePolicy);
                    break;
                case Step:
                case Sigmoid:
                    if (Double.isNaN(lrPolicySteps))
                        throw new IllegalStateException("Layer \"" + layerName
                                        + "\" learning rate policy steps (lrPolicySteps) must be set to use "
                                        + learningRatePolicy);
                    break;
                case Schedule:
                    if (learningRateSchedule == null)
                        throw new IllegalStateException("Layer \"" + layerName
                                        + "\" learning rate policy schedule (learningRateSchedule) must be set to use "
                                        + learningRatePolicy);
                    break;
            }

            if (!Double.isNaN(lrPolicyPower) && (learningRatePolicy != LearningRatePolicy.Inverse
                            && learningRatePolicy != LearningRatePolicy.Poly))
                throw new IllegalStateException("Layer \"" + layerName
                                + "\" power has been set but will not be applied unless the learning rate policy is set to Inverse or Poly.");
            if (!Double.isNaN(lrPolicySteps) && (learningRatePolicy != LearningRatePolicy.Step
                            && learningRatePolicy != LearningRatePolicy.Sigmoid
                            && learningRatePolicy != LearningRatePolicy.TorchStep))
                throw new IllegalStateException("Layer \"" + layerName
                                + "\" steps have been set but will not be applied unless the learning rate policy is set to Step or Sigmoid.");
            if ((learningRateSchedule != null) && (learningRatePolicy != LearningRatePolicy.Schedule))
                throw new IllegalStateException("Layer \"" + layerName
                                + "\" learning rate schedule has been set but will not be applied unless the learning rate policy is set to Schedule.");

        }
        ////////////////

        /**
         * Return a configuration based on this builder
         *
         * @return
         */
        public NeuralNetConfiguration build() {

            NeuralNetConfiguration conf = new NeuralNetConfiguration();
            conf.minimize = minimize;
            conf.maxNumLineSearchIterations = maxNumLineSearchIterations;
            conf.layer = layer;
            conf.numIterations = numIterations;
            conf.useRegularization = useRegularization;
            conf.optimizationAlgo = optimizationAlgo;
            conf.seed = seed;
            conf.stepFunction = stepFunction;
            conf.useDropConnect = useDropConnect;
            conf.miniBatch = miniBatch;
            conf.learningRatePolicy = learningRatePolicy;
            conf.lrPolicyDecayRate = lrPolicyDecayRate;
            conf.lrPolicySteps = lrPolicySteps;
            conf.lrPolicyPower = lrPolicyPower;
            conf.pretrain = pretrain;
            String layerName;
            if (layer == null || layer.getLayerName() == null)
                layerName = "Layer not named";
            else
                layerName = layer.getLayerName();
            learningRateValidation(layerName);

            if (layer != null) {
                if (Double.isNaN(layer.getLearningRate()))
                    layer.setLearningRate(learningRate);
                if (Double.isNaN(layer.getBiasLearningRate()))
                    layer.setBiasLearningRate(layer.getLearningRate());
                if (layer.getLearningRateSchedule() == null)
                    layer.setLearningRateSchedule(learningRateSchedule);
                if (Double.isNaN(layer.getL1()))
                    layer.setL1(l1);
                if (Double.isNaN(layer.getL2()))
                    layer.setL2(l2);
                if (layer.getActivationFn() == null)
                    layer.setActivationFn(activationFn);
                if (layer.getWeightInit() == null)
                    layer.setWeightInit(weightInit);
                if (Double.isNaN(layer.getBiasInit()))
                    layer.setBiasInit(biasInit);
                if (Double.isNaN(layer.getDropOut()))
                    layer.setDropOut(dropOut);
                if (layer.getUpdater() == null)
                    layer.setUpdater(updater);
                //                updaterValidation(layerName);
                LayerValidation.updaterValidation(layerName, layer, momentum, momentumSchedule, adamMeanDecay,
                                adamVarDecay, rho, rmsDecay, epsilon);
                if (layer.getGradientNormalization() == null)
                    layer.setGradientNormalization(gradientNormalization);
                if (Double.isNaN(layer.getGradientNormalizationThreshold()))
                    layer.setGradientNormalizationThreshold(gradientNormalizationThreshold);
                if (layer instanceof ConvolutionLayer) {
                    ConvolutionLayer cl = (ConvolutionLayer) layer;
                    if (cl.getConvolutionMode() == null) {
                        cl.setConvolutionMode(convolutionMode);
                    }
                }
                if (layer instanceof SubsamplingLayer) {
                    SubsamplingLayer sl = (SubsamplingLayer) layer;
                    if (sl.getConvolutionMode() == null) {
                        sl.setConvolutionMode(convolutionMode);
                    }
                }
            }
            LayerValidation.generalValidation(layerName, layer, useRegularization, useDropConnect, dropOut, l2, l2Bias,
                            l1, l1Bias, dist);
            return conf;
        }

    }
}
