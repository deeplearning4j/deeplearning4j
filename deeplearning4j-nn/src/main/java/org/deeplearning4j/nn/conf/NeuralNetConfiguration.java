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
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ClassUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution;
import org.deeplearning4j.nn.conf.serde.ComputationGraphConfigurationDeserializer;
import org.deeplearning4j.nn.conf.serde.MultiLayerConfigurationDeserializer;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.reflections.DL4JSubTypesScanner;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.shade.jackson.databind.*;
import org.nd4j.shade.jackson.databind.deser.BeanDeserializerModifier;
import org.nd4j.shade.jackson.databind.introspect.AnnotatedClass;
import org.nd4j.shade.jackson.databind.jsontype.NamedType;
import org.nd4j.shade.jackson.databind.module.SimpleModule;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import org.reflections.ReflectionUtils;
import org.reflections.Reflections;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;
import org.reflections.util.FilterBuilder;

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
@Slf4j
public class NeuralNetConfiguration implements Serializable, Cloneable {


    /**
     * System property for custom layers, preprocessors, graph vertices etc. Enabled by default.
     * Run JVM with "-Dorg.deeplearning4j.config.custom.enabled=false" to disable classpath scanning for
     * Overriding the default (i.e., disabling) this is only useful if (a) no custom layers/preprocessors etc will be
     * used, and (b) minimizing startup/initialization time for new JVMs is very important.
     * Results are cached, so there is no cost to custom layers after the first network has been constructed.
     */
    public static final String CUSTOM_FUNCTIONALITY = "org.deeplearning4j.config.custom.enabled";

    protected Layer layer;
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
    protected StepFunction stepFunction;
    //minimize or maximize objective
    protected boolean minimize = true;
    protected Map<String, Double> l1ByParam = new HashMap<>();
    protected Map<String, Double> l2ByParam = new HashMap<>();
    protected boolean pretrain;

    // this field defines preOutput cache
    protected CacheMode cacheMode;

    //Counter for the number of parameter updates so far for this layer.
    //Note that this is only used for pretrain layers (RBM, VAE) - MultiLayerConfiguration and ComputationGraphConfiguration
    //contain counters for standard backprop training.
    // This is important for learning rate schedules, for example, and is stored here to ensure it is persisted
    // for Spark and model serialization
    protected int iterationCount = 0;

    //Counter for the number of epochs completed so far. Used for per-epoch schedules
    protected int epochCount = 0;

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
        l1ByParam.clear();
        l2ByParam.clear();
    }

    public void resetVariables() {
        for (String s : variables) {
            setLayerParamLR(s);
        }
    }

    public void setLayerParamLR(String variable) {
        double l1 = layer.getL1ByParam(variable);
        if (Double.isNaN(l1))
            l1 = 0.0; //Not set
        double l2 = layer.getL2ByParam(variable);
        if (Double.isNaN(l2))
            l2 = 0.0; //Not set
        l1ByParam.put(variable, l1);
        l2ByParam.put(variable, l2);
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
        private int layerCounter = -1; //Used only for .layer(Layer) method
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

        public ListBuilder layer(int ind, @NonNull Layer layer) {
            if (layerwise.containsKey(ind)) {
                log.info("Layer index {} already exists, layer of type {} will be replace by layer type {}",
                        ind, layerwise.get(ind).getClass().getSimpleName(), layer.getClass().getSimpleName());
                layerwise.get(ind).layer(layer);
            } else {
                layerwise.put(ind, globalConfig.clone().layer(layer));
            }
            if(layerCounter < ind){
                //Edge case: user is mixing .layer(Layer) and .layer(int, Layer) calls
                //This should allow a .layer(A, X) and .layer(Y) to work such that layer Y is index (A+1)
                layerCounter = ind;
            }
            return this;
        }

        public ListBuilder layer(Layer layer){
            return layer(++layerCounter, layer);
        }

        public Map<Integer, Builder> getLayerwise() {
            return layerwise;
        }

        @Override
        public ListBuilder setInputType(InputType inputType){
            return (ListBuilder)super.setInputType(inputType);
        }

        /**
         * A convenience method for setting input types: note that for example .inputType().convolutional(h,w,d)
         * is equivalent to .setInputType(InputType.convolutional(h,w,d))
         */
        public ListBuilder.InputTypeBuilder inputType(){
            return new InputTypeBuilder();
        }

        /**
         * Build the multi layer network
         * based on this neural network and
         * overr ridden parameters
         *
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
                            .tBPTTBackwardLength(tbpttBackLength).setInputType(this.inputType)
                            .trainingWorkspaceMode(globalConfig.trainingWorkspaceMode).cacheMode(globalConfig.cacheMode)
                            .inferenceWorkspaceMode(globalConfig.inferenceWorkspaceMode).confs(list).build();
        }

        /** Helper class for setting input types */
        public class InputTypeBuilder {
            /**
             * See {@link InputType#convolutional(int, int, int)}
             */
            public ListBuilder convolutional(int height, int width, int depth){
                return ListBuilder.this.setInputType(InputType.convolutional(height, width, depth));
            }

            /**
             * * See {@link InputType#convolutionalFlat(int, int, int)}
             */
            public ListBuilder convolutionalFlat(int height, int width, int depth){
                return ListBuilder.this.setInputType(InputType.convolutionalFlat(height, width, depth));
            }

            /**
             * See {@link InputType#feedForward(int)}
             */
            public ListBuilder feedForward(int size){
                return ListBuilder.this.setInputType(InputType.feedForward(size));
            }

            /**
             * See {@link InputType#recurrent(int)}}
             */
            public ListBuilder recurrent(int size){
                return ListBuilder.this.setInputType(InputType.recurrent(size));
            }
        }
    }

    /**
     * Return this configuration as json
     *
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
     *
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
     *
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
     *
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
     *
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
     *
     * @return
     */
    public static ObjectMapper mapper() {
        return mapper;
    }

    /**
     * Reinitialize and return the Jackson/json ObjectMapper with additional named types.
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

        SimpleModule customDeserializerModule = new SimpleModule();
        customDeserializerModule.setDeserializerModifier(new BeanDeserializerModifier() {
            @Override
            public JsonDeserializer<?> modifyDeserializer(DeserializationConfig config, BeanDescription beanDesc,
                            JsonDeserializer<?> deserializer) {
                //Use our custom deserializers to handle backward compatibility for updaters -> IUpdater
                if (beanDesc.getBeanClass() == MultiLayerConfiguration.class) {
                    return new MultiLayerConfigurationDeserializer(deserializer);
                } else if (beanDesc.getBeanClass() == ComputationGraphConfiguration.class) {
                    return new ComputationGraphConfigurationDeserializer(deserializer);
                }
                return deserializer;
            }
        });

        ret.registerModule(customDeserializerModule);

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
        protected double l1 = Double.NaN;
        protected double l2 = Double.NaN;
        protected double l1Bias = Double.NaN;
        protected double l2Bias = Double.NaN;
        protected IDropout idropOut;
        protected IWeightNoise weightNoise;
        protected IUpdater iUpdater = new Sgd();
        protected IUpdater biasUpdater = null;
        protected Layer layer;
        protected boolean miniBatch = true;
        protected int numIterations = 1;
        protected int maxNumLineSearchIterations = 5;
        protected long seed = System.currentTimeMillis();
        protected OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
        protected StepFunction stepFunction = null;
        protected boolean minimize = true;
        protected GradientNormalization gradientNormalization = GradientNormalization.None;
        protected double gradientNormalizationThreshold = 1.0;
        protected boolean pretrain = false;
        protected List<LayerConstraint> allParamConstraints;
        protected List<LayerConstraint> weightConstraints;
        protected List<LayerConstraint> biasConstraints;

        protected WorkspaceMode trainingWorkspaceMode = WorkspaceMode.NONE;
        protected WorkspaceMode inferenceWorkspaceMode = WorkspaceMode.SEPARATE;
        protected CacheMode cacheMode = CacheMode.NONE;

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
                optimizationAlgo = newConf.optimizationAlgo;
                seed = newConf.seed;
                stepFunction = newConf.stepFunction;
                miniBatch = newConf.miniBatch;
                pretrain = newConf.pretrain;
            }
        }

        /**
         * Process input as minibatch vs full dataset.
         * Default set to true.
         */
        public Builder miniBatch(boolean miniBatch) {
            this.miniBatch = miniBatch;
            return this;
        }

        /**
         * This method defines Workspace mode being used during training:
         * NONE: workspace won't be used
         * SINGLE: one workspace will be used during whole iteration loop
         * SEPARATE: separate workspaces will be used for feedforward and backprop iteration loops
         *
         * @param workspaceMode
         * @return
         */
        public Builder trainingWorkspaceMode(@NonNull WorkspaceMode workspaceMode) {
            this.trainingWorkspaceMode = workspaceMode;
            return this;
        }

        /**
         * This method defines Workspace mode being used during inference:
         * NONE: workspace won't be used
         * SINGLE: one workspace will be used during whole iteration loop
         * SEPARATE: separate workspaces will be used for feedforward and backprop iteration loops
         *
         * @param workspaceMode
         * @return
         */
        public Builder inferenceWorkspaceMode(@NonNull WorkspaceMode workspaceMode) {
            this.inferenceWorkspaceMode = workspaceMode;
            return this;
        }

        /**
         * This method defines how/if preOutput cache is handled:
         * NONE: cache disabled (default value)
         * HOST: Host memory will be used
         * DEVICE: GPU memory will be used (on CPU backends effect will be the same as for HOST)
         *
         * @param cacheMode
         * @return
         */
        public Builder cacheMode(@NonNull CacheMode cacheMode) {
            this.cacheMode = cacheMode;
            return this;
        }

        /**
         * Objective function to minimize or maximize cost function
         * Default set to minimize true.
         */
        public Builder minimize(boolean minimize) {
            this.minimize = minimize;
            return this;
        }

        /**
         * Maximum number of line search iterations.
         * Only applies for line search optimizers: Line Search SGD, Conjugate Gradient, LBFGS
         * is NOT applicable for standard SGD
         *
         * @param maxNumLineSearchIterations > 0
         * @return
         */
        public Builder maxNumLineSearchIterations(int maxNumLineSearchIterations) {
            this.maxNumLineSearchIterations = maxNumLineSearchIterations;
            return this;
        }


        /**
         * Layer class.
         */
        public Builder layer(Layer layer) {
            this.layer = layer;
            return this;
        }

        /**
         * Step function to apply for back track line search.
         * Only applies for line search optimizers: Line Search SGD, Conjugate Gradient, LBFGS
         * Options: DefaultStepFunction (default), NegativeDefaultStepFunction
         * GradientStepFunction (for SGD), NegativeGradientStepFunction
         */
        @Deprecated
        public Builder stepFunction(StepFunction stepFunction) {
            this.stepFunction = stepFunction;
            return this;
        }

        /**
         * Create a ListBuilder (for creating a MultiLayerConfiguration)<br>
         * Usage:<br>
         * <pre>
         * {@code .list()
         * .layer(new DenseLayer.Builder()...build())
         * ...
         * .layer(new OutputLayer.Builder()...build())
         * }
         * </pre>
         */
        public ListBuilder list() {
            return new ListBuilder(this);
        }

        /**
         * Create a ListBuilder (for creating a MultiLayerConfiguration) with the specified layers<br>
         * Usage:<br>
         * <pre>
         * {@code .list(
         *      new DenseLayer.Builder()...build(),
         *      ...,
         *      new OutputLayer.Builder()...build())
         * }
         * </pre>
         *
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

        /**
         * Number of optimization iterations. Should be set to 1 for >99% of use cases (possible exception:
         * very tiny full batch dataset training)
         */
        public Builder iterations(int numIterations) {
            this.numIterations = numIterations;
            return this;
        }

        /**
         * Random number generator seed. Used for reproducability between runs
         */
        public Builder seed(long seed) {
            this.seed = seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }

        /**
         * Optimization algorithm to use. Most common: OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
         *
         * @param optimizationAlgo Optimization algorithm to use when training
         */
        public Builder optimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
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

        /**
         * Activation function / neuron non-linearity
         *
         * @see #activation(Activation)
         */
        public Builder activation(IActivation activationFunction) {
            this.activationFn = activationFunction;
            return this;
        }

        /**
         * Activation function / neuron non-linearity
         */
        public Builder activation(Activation activation) {
            return activation(activation.getActivationFunction());
        }

        /**
         * Weight initialization scheme.
         *
         * @see org.deeplearning4j.nn.weights.WeightInit
         */
        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }

        /**
         * Constant for bias initialization. Default: 0.0
         *
         * @param biasInit Constant for bias initialization
         */
        public Builder biasInit(double biasInit) {
            this.biasInit = biasInit;
            return this;
        }

        /**
         * Distribution to sample initial weights from. Used in conjunction with
         * .weightInit(WeightInit.DISTRIBUTION).
         */
        public Builder dist(Distribution dist) {
            this.dist = dist;
            return this;
        }

        /**
         * L1 regularization coefficient for the weights.
         */
        public Builder l1(double l1) {
            this.l1 = l1;
            return this;
        }

        /**
         * L2 regularization coefficient for the weights.
         */
        public Builder l2(double l2) {
            this.l2 = l2;
            return this;
        }

        /**
         * L1 regularization coefficient for the bias.
         */
        public Builder l1Bias(double l1Bias) {
            this.l1Bias = l1Bias;
            return this;
        }

        /**
         * L2 regularization coefficient for the bias.
         */
        public Builder l2Bias(double l2Bias) {
            this.l2Bias = l2Bias;
            return this;
        }

        /**
         * Dropout probability. This is the probability of <it>retaining</it> each input activation value for a layer.
         * dropOut(x) will keep an input activation with probability x, and set to 0 with probability 1-x.<br>
         * dropOut(0.0) is a special value / special case - when set to 0.0., dropout is disabled (not applied). Note
         * that a dropout value of 1.0 is functionally equivalent to no dropout: i.e., 100% probability of retaining
         * each input activation.<br>
         * <p>
         * Note 1: Dropout is applied at training time only - and is automatically not applied at test time
         * (for evaluation, etc)<br>
         * Note 2: This sets the probability per-layer. Care should be taken when setting lower values for
         * complex networks (too much information may be lost with aggressive (very low) dropout values).<br>
         * Note 3: Frequently, dropout is not applied to (or, has higher retain probability for) input (first layer)
         * layers. Dropout is also often not applied to output layers. This needs to be handled MANUALLY by the user
         * - set .dropout(0) on those layers when using global dropout setting.<br>
         * Note 4: Implementation detail (most users can ignore): DL4J uses inverted dropout, as described here:
         * <a href="http://cs231n.github.io/neural-networks-2/">http://cs231n.github.io/neural-networks-2/</a>
         * </p>
         *
         * @param inputRetainProbability Dropout probability (probability of retaining each input activation value for a layer)
         * @see #dropOut(IDropout)
         */
        public Builder dropOut(double inputRetainProbability) {
            return dropOut(new Dropout(inputRetainProbability));
        }

        /**
         * Set the dropout for all layers in this network
         *
         * @param dropout Dropout, such as {@link Dropout}, {@link org.deeplearning4j.nn.conf.dropout.GaussianDropout},
         *                {@link org.deeplearning4j.nn.conf.dropout.GaussianNoise} etc
         * @return
         */
        public Builder dropOut(IDropout dropout){
            this.idropOut = dropout;
            return this;
        }

        /**
         * Set the weight noise (such as {@link org.deeplearning4j.nn.conf.weightnoise.DropConnect} and
         * {@link org.deeplearning4j.nn.conf.weightnoise.WeightNoise}) for the layers in this network.
         *
         * @param weightNoise Weight noise instance to use
         */
        public Builder weightNoise(IWeightNoise weightNoise){
            this.weightNoise = weightNoise;
            return this;
        }


        /**
         * @deprecated Use {@link #updater(IUpdater)}
         */
        @Deprecated
        public Builder updater(Updater updater) {
            return updater(updater.getIUpdaterWithDefaultConfig());
        }

        /**
         * Gradient updater configuration. For example, {@link org.nd4j.linalg.learning.config.Adam}
         * or {@link org.nd4j.linalg.learning.config.Nesterovs}
         *
         * @param updater Updater to use
         */
        public Builder updater(IUpdater updater) {
            this.iUpdater = updater;
            return this;
        }

        /**
         * Gradient updater configuration, for the biases only. If not set, biases will use the updater as
         * set by {@link #updater(IUpdater)}
         *
         * @param updater Updater to use for bias parameters
         */
        public Builder biasUpdater(IUpdater updater){
            this.biasUpdater = updater;
            return this;
        }

        /**
         * Gradient normalization strategy. Used to specify gradient renormalization, gradient clipping etc.
         * See {@link GradientNormalization} for details
         *
         * @param gradientNormalization Type of normalization to use. Defaults to None.
         * @see GradientNormalization
         */
        public Builder gradientNormalization(GradientNormalization gradientNormalization) {
            this.gradientNormalization = gradientNormalization;
            return this;
        }

        /**
         * Threshold for gradient normalization, only used for GradientNormalization.ClipL2PerLayer,
         * GradientNormalization.ClipL2PerParamType, and GradientNormalization.ClipElementWiseAbsoluteValue<br>
         * Not used otherwise.<br>
         * L2 threshold for first two types of clipping, or absolute value threshold for last type of clipping.
         */
        public Builder gradientNormalizationThreshold(double threshold) {
            this.gradientNormalizationThreshold = threshold;
            return this;
        }

        /**
         * Sets the convolution mode for convolutional layers, which impacts padding and output sizes.
         * See {@link ConvolutionMode} for details. Defaults to ConvolutionMode.TRUNCATE
         * @param convolutionMode Convolution mode to use
         */
        public Builder convolutionMode(ConvolutionMode convolutionMode) {
            this.convolutionMode = convolutionMode;
            return this;
        }

        /**
         * Set constraints to be applied to all layers. Default: no constraints.<br>
         * Constraints can be used to enforce certain conditions (non-negativity of parameters, max-norm regularization,
         * etc). These constraints are applied at each iteration, after the parameters have been updated.
         *
         * @param constraints Constraints to apply to all parameters of all layers
         */
        public Builder constrainAllParameters(LayerConstraint... constraints){
            this.allParamConstraints = Arrays.asList(constraints);
            return this;
        }

        /**
         * Set constraints to be applied to all layers. Default: no constraints.<br>
         * Constraints can be used to enforce certain conditions (non-negativity of parameters, max-norm regularization,
         * etc). These constraints are applied at each iteration, after the parameters have been updated.
         *
         * @param constraints Constraints to apply to all bias parameters of all layers
         */
        public Builder constrainBias(LayerConstraint... constraints) {
            this.biasConstraints = Arrays.asList(constraints);
            return this;
        }

        /**
         * Set constraints to be applied to all layers. Default: no constraints.<br>
         * Constraints can be used to enforce certain conditions (non-negativity of parameters, max-norm regularization,
         * etc). These constraints are applied at each iteration, after the parameters have been updated.
         *
         * @param constraints Constraints to apply to all weight parameters of all layers
         */
        public Builder constrainWeights(LayerConstraint... constraints) {
            this.weightConstraints = Arrays.asList(constraints);
            return this;
        }

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
            conf.optimizationAlgo = optimizationAlgo;
            conf.seed = seed;
            conf.stepFunction = stepFunction;
            conf.miniBatch = miniBatch;
            conf.pretrain = pretrain;
            conf.cacheMode = this.cacheMode;

            configureLayer(layer);
            if (layer instanceof FrozenLayer) {
                configureLayer(((FrozenLayer) layer).getLayer());
            }

            return conf;
        }

        private void configureLayer(Layer layer) {
            String layerName;
            if (layer == null || layer.getLayerName() == null)
                layerName = "Layer not named";
            else
                layerName = layer.getLayerName();

            if (layer != null) {
                copyConfigToLayer(layerName, layer);
            }

            if (layer instanceof FrozenLayer) {
                copyConfigToLayer(layerName, ((FrozenLayer) layer).getLayer());
            }

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
            LayerValidation.generalValidation(layerName, layer, idropOut, l2, l2Bias, l1, l1Bias, dist,
                    allParamConstraints, weightConstraints, biasConstraints);
        }

        private void copyConfigToLayer(String layerName, Layer layer) {

            if (layer.getIDropout() == null)
                layer.setIDropout(idropOut);

            if (layer instanceof BaseLayer) {
                BaseLayer bLayer = (BaseLayer) layer;
                if (Double.isNaN(bLayer.getL1()))
                    bLayer.setL1(l1);
                if (Double.isNaN(bLayer.getL2()))
                    bLayer.setL2(l2);
                if (bLayer.getActivationFn() == null)
                    bLayer.setActivationFn(activationFn);
                if (bLayer.getWeightInit() == null)
                    bLayer.setWeightInit(weightInit);
                if (Double.isNaN(bLayer.getBiasInit()))
                    bLayer.setBiasInit(biasInit);

                //Configure weight noise:
                if(weightNoise != null && ((BaseLayer) layer).getWeightNoise() == null){
                    ((BaseLayer) layer).setWeightNoise(weightNoise.clone());
                }

                //Configure updaters:
                if(iUpdater != null && bLayer.getIUpdater() == null){
                    bLayer.setIUpdater(iUpdater);
                }
                if(biasUpdater != null && bLayer.getBiasUpdater() == null){
                    bLayer.setBiasUpdater(biasUpdater);
                }

                if(bLayer.getIUpdater() == null && iUpdater == null && bLayer.initializer().numParams(bLayer) > 0){
                    //No updater set anywhere
                    IUpdater u = new Sgd();
                    bLayer.setIUpdater(u);
                    log.warn("*** No updater configuration is set for layer {} - defaulting to {} ***", layerName, u);
                }

                if (bLayer.getGradientNormalization() == null)
                    bLayer.setGradientNormalization(gradientNormalization);
                if (Double.isNaN(bLayer.getGradientNormalizationThreshold()))
                    bLayer.setGradientNormalizationThreshold(gradientNormalizationThreshold);
            }
        }
    }

}
