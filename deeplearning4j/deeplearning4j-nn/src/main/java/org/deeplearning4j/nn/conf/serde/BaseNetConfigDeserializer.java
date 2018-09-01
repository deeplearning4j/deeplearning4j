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

import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonMappingException;
import org.nd4j.shade.jackson.databind.deser.ResolvableDeserializer;
import org.nd4j.shade.jackson.databind.deser.std.StdDeserializer;
import org.nd4j.shade.jackson.databind.node.ObjectNode;

import java.io.IOException;

/**
 * A custom (abstract) deserializer that handles backward compatibility (currently only for updater refactoring that
 * happened after 0.8.0). This is used for both MultiLayerConfiguration and ComputationGraphConfiguration.<br>
 * We deserialize the config using the default deserializer, then handle the new IUpdater (which will be null for
 * 0.8.0 and earlier configs) if necessary
 *
 * Overall design: <a href="http://stackoverflow.com/questions/18313323/how-do-i-call-the-default-deserializer-from-a-custom-deserializer-in-jackson">
 *     http://stackoverflow.com/questions/18313323/how-do-i-call-the-default-deserializer-from-a-custom-deserializer-in-jackson</a>
 *
 * @author Alex Black
 */
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

    @Override
    public void resolve(DeserializationContext ctxt) throws JsonMappingException {
        ((ResolvableDeserializer) defaultDeserializer).resolve(ctxt);
    }
}
