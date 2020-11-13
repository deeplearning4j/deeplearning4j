/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.conf.serde;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.*;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.learning.regularization.WeightDecay;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.*;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonMappingException;
import org.nd4j.shade.jackson.databind.deser.ResolvableDeserializer;
import org.nd4j.shade.jackson.databind.deser.std.StdDeserializer;
import org.nd4j.shade.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * A custom (abstract) deserializer that handles backward compatibility (currently only for updater refactoring that
 * happened after 0.8.0). This is used for both MultiLayerConfiguration and ComputationGraphConfiguration.<br>
 * We deserialize the config using the default deserializer, then handle the new IUpdater (which will be null for
 * 0.8.0 and earlier configs) if necessary
 *
 * Overall design: <a href="https://stackoverflow.com/questions/18313323/how-do-i-call-the-default-deserializer-from-a-custom-deserializer-in-jackson">
 *     https://stackoverflow.com/questions/18313323/how-do-i-call-the-default-deserializer-from-a-custom-deserializer-in-jackson</a>
 *
 * @author Alex Black
 */
@Slf4j
public abstract class BaseNetConfigDeserializer<T> extends StdDeserializer<T> implements ResolvableDeserializer {

    protected final JsonDeserializer<?> defaultDeserializer;

    public BaseNetConfigDeserializer(JsonDeserializer<?> defaultDeserializer, Class<T> deserializedType) {
        super(deserializedType);
        this.defaultDeserializer = defaultDeserializer;
    }

    @Override
    public abstract T deserialize(JsonParser jp, DeserializationContext ctxt)
                    throws IOException, JsonProcessingException;

    protected boolean requiresIUpdaterFromLegacy(Layer[] layers){
        for(Layer l : layers){
            if(l instanceof BaseLayer){
                BaseLayer bl = (BaseLayer)l;
                if(bl.getIUpdater() == null && bl.initializer().numParams(bl) > 0){
                    return true;
                }
            }
        }
        return false;
    }

    protected boolean requiresDropoutFromLegacy(Layer[] layers){
        for(Layer l : layers){
            if(l.getIDropout() != null){
                return false;
            }
        }
        return true;
    }

    protected boolean requiresRegularizationFromLegacy(Layer[] layers){
        for(Layer l : layers){
            if(l instanceof BaseLayer && ((BaseLayer)l).getRegularization() == null){
                return true;
            }
        }
        return false;
    }

    protected boolean requiresWeightInitFromLegacy(Layer[] layers){
        for(Layer l : layers){
            if(l instanceof BaseLayer && ((BaseLayer)l).getWeightInitFn() == null){
                return true;
            }
        }
        return false;
    }

    protected boolean requiresActivationFromLegacy(Layer[] layers){
        for(Layer l : layers){
            if(l instanceof BaseLayer && ((BaseLayer)l).getActivationFn() == null){
                return true;
            }
        }
        return false;
    }

    protected boolean requiresLegacyLossHandling(Layer[] layers){
        for(Layer l : layers){
            if(l instanceof BaseOutputLayer && ((BaseOutputLayer)l).getLossFn() == null){
                return true;
            }
        }
        return false;
    }

    protected void handleUpdaterBackwardCompatibility(BaseLayer layer, ObjectNode on){
        if(on != null && on.has("updater")){
            String updaterName = on.get("updater").asText();
            if(updaterName != null){
                Updater u = Updater.valueOf(updaterName);
                IUpdater iu = u.getIUpdaterWithDefaultConfig();
                double lr = on.get("learningRate").asDouble();
                double eps;
                if(on.has("epsilon")){
                    eps = on.get("epsilon").asDouble();
                } else {
                    eps = Double.NaN;
                }
                double rho = on.get("rho").asDouble();
                switch (u){
                    case SGD:
                        ((Sgd)iu).setLearningRate(lr);
                        break;
                    case ADAM:
                        if(Double.isNaN(eps)){
                            eps = Adam.DEFAULT_ADAM_EPSILON;
                        }
                        ((Adam)iu).setLearningRate(lr);
                        ((Adam)iu).setBeta1(on.get("adamMeanDecay").asDouble());
                        ((Adam)iu).setBeta2(on.get("adamVarDecay").asDouble());
                        ((Adam)iu).setEpsilon(eps);
                        break;
                    case ADAMAX:
                        if(Double.isNaN(eps)){
                            eps = AdaMax.DEFAULT_ADAMAX_EPSILON;
                        }
                        ((AdaMax)iu).setLearningRate(lr);
                        ((AdaMax)iu).setBeta1(on.get("adamMeanDecay").asDouble());
                        ((AdaMax)iu).setBeta2(on.get("adamVarDecay").asDouble());
                        ((AdaMax)iu).setEpsilon(eps);
                        break;
                    case ADADELTA:
                        if(Double.isNaN(eps)){
                            eps = AdaDelta.DEFAULT_ADADELTA_EPSILON;
                        }
                        ((AdaDelta)iu).setRho(rho);
                        ((AdaDelta)iu).setEpsilon(eps);
                        break;
                    case NESTEROVS:
                        ((Nesterovs)iu).setLearningRate(lr);
                        ((Nesterovs)iu).setMomentum(on.get("momentum").asDouble());
                        break;
                    case NADAM:
                        if(Double.isNaN(eps)){
                            eps = Nadam.DEFAULT_NADAM_EPSILON;
                        }
                        ((Nadam)iu).setLearningRate(lr);
                        ((Nadam)iu).setBeta1(on.get("adamMeanDecay").asDouble());
                        ((Nadam)iu).setBeta2(on.get("adamVarDecay").asDouble());
                        ((Nadam)iu).setEpsilon(eps);
                        break;
                    case ADAGRAD:
                        if(Double.isNaN(eps)){
                            eps = AdaGrad.DEFAULT_ADAGRAD_EPSILON;
                        }
                        ((AdaGrad)iu).setLearningRate(lr);
                        ((AdaGrad)iu).setEpsilon(eps);
                        break;
                    case RMSPROP:
                        if(Double.isNaN(eps)){
                            eps = RmsProp.DEFAULT_RMSPROP_EPSILON;
                        }
                        ((RmsProp)iu).setLearningRate(lr);
                        ((RmsProp)iu).setEpsilon(eps);
                        ((RmsProp)iu).setRmsDecay(on.get("rmsDecay").asDouble());
                        break;
                    default:
                        //No op
                        break;
                }

                layer.setIUpdater(iu);
            }
        }
    }

    protected void handleL1L2BackwardCompatibility(BaseLayer baseLayer, ObjectNode on){
        if(on != null && (on.has("l1") || on.has("l2"))){
            //Legacy format JSON
            baseLayer.setRegularization(new ArrayList<Regularization>());
            baseLayer.setRegularizationBias(new ArrayList<Regularization>());

            if(on.has("l1")){
                double l1 = on.get("l1").doubleValue();
                if(l1 > 0.0){
                    baseLayer.getRegularization().add(new L1Regularization(l1));
                }
            }
            if(on.has("l2")){
                double l2 = on.get("l2").doubleValue();
                if(l2 > 0.0){
                    //Default to non-LR based WeightDecay, to match behaviour in 1.0.0-beta3
                    baseLayer.getRegularization().add(new WeightDecay(l2, false));
                }
            }
            if(on.has("l1Bias")){
                double l1Bias = on.get("l1Bias").doubleValue();
                if(l1Bias > 0.0){
                    baseLayer.getRegularizationBias().add(new L1Regularization(l1Bias));
                }
            }
            if(on.has("l2Bias")){
                double l2Bias = on.get("l2Bias").doubleValue();
                if(l2Bias > 0.0){
                    //Default to non-LR based WeightDecay, to match behaviour in 1.0.0-beta3
                    baseLayer.getRegularizationBias().add(new WeightDecay(l2Bias, false));
                }
            }
        }
    }

    protected void handleWeightInitBackwardCompatibility(BaseLayer baseLayer, ObjectNode on){
        if(on != null && on.has("weightInit") ){
            //Legacy format JSON
            if(on.has("weightInit")){
                String wi = on.get("weightInit").asText();
                try{
                    WeightInit w = WeightInit.valueOf(wi);
                    Distribution d = null;
                    if(w == WeightInit.DISTRIBUTION && on.has("dist")){
                        String dist = on.get("dist").toString();
                        d = NeuralNetConfiguration.mapper().readValue(dist, Distribution.class);
                    }
                    IWeightInit iwi = w.getWeightInitFunction(d);
                    baseLayer.setWeightInitFn(iwi);
                } catch (Throwable t){
                    log.warn("Failed to infer weight initialization from legacy JSON format",t);
                }
            }
        }
    }

    //Changed after 0.7.1 from "activationFunction" : "softmax" to "activationFn" : <object>
    protected void handleActivationBackwardCompatibility(BaseLayer baseLayer, ObjectNode on){
        if(baseLayer.getActivationFn() == null && on.has("activationFunction")){
            String afn = on.get("activationFunction").asText();
            IActivation a = null;
            try {
                a = getMap()
                        .get(afn.toLowerCase())
                        .getDeclaredConstructor()
                        .newInstance();
            } catch (InstantiationException | IllegalAccessException | NoSuchMethodException
                    | InvocationTargetException instantiationException){
                log.error(instantiationException.getMessage());
            }
            baseLayer.setActivationFn(a);
        }
    }

    //0.5.0 and earlier: loss function was an enum like "lossFunction" : "NEGATIVELOGLIKELIHOOD",
    protected void handleLossBackwardCompatibility(BaseOutputLayer baseLayer, ObjectNode on){
        if(baseLayer.getLossFn() == null && on.has("activationFunction")) {
            String lfn = on.get("lossFunction").asText();
            ILossFunction loss = null;
            switch (lfn) {
                case "MCXENT":
                    loss = new LossMCXENT();
                    break;
                case "MSE":
                    loss = new LossMSE();
                    break;
                case "NEGATIVELOGLIKELIHOOD":
                    loss = new LossNegativeLogLikelihood();
                    break;
                case "SQUARED_LOSS":
                    loss = new LossL2();
                    break;
                case "XENT":
                    loss = new LossBinaryXENT();
            }
            baseLayer.setLossFn(loss);
        }
    }

    private static Map<String,Class<? extends IActivation>> activationMap;
    private static synchronized Map<String,Class<? extends IActivation>> getMap(){
        if(activationMap == null){
            activationMap = new HashMap<>();
            for(Activation a : Activation.values()){
                activationMap.put(a.toString().toLowerCase(), a.getActivationFunction().getClass());
            }
        }
        return activationMap;
    }

    @Override
    public void resolve(DeserializationContext ctxt) throws JsonMappingException {
        ((ResolvableDeserializer) defaultDeserializer).resolve(ctxt);
    }
}
