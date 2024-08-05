/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.interceptor;

import net.bytebuddy.agent.builder.AgentBuilder;
import net.bytebuddy.asm.Advice;
import net.bytebuddy.description.type.TypeDescription;
import net.bytebuddy.dynamic.ClassFileLocator;
import net.bytebuddy.dynamic.DynamicType;
import net.bytebuddy.implementation.MethodDelegation;
import net.bytebuddy.matcher.ElementMatchers;
import net.bytebuddy.utility.JavaModule;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.interceptor.advice.MultiLayerNetworkBackwardAdvice;
import org.nd4j.interceptor.advice.MultiLayerNetworkForwardAdvice;
import org.nd4j.interceptor.data.InterceptorPersistence;
import org.nd4j.interceptor.transformers.*;
import org.nd4j.interceptor.util.InterceptorUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;

import java.io.File;
import java.lang.instrument.Instrumentation;
import java.security.ProtectionDomain;

import static net.bytebuddy.matcher.ElementMatchers.none;

public class Nd4jInterceptor {


    public static void premain(String agentArgs, Instrumentation inst) {
        AgentBuilder agentBuilder =  new AgentBuilder.Default()
                .ignore(none())
                .with(AgentBuilder.RedefinitionStrategy.RETRANSFORMATION)
                .type(ElementMatchers.nameContains("MultiLayerNetwork"))
                .transform(new MultiLayerNetworkTransformer());


        agentBuilder.installOn(inst);


        AgentBuilder agentBuilder6 =  new AgentBuilder.Default()
                .ignore(none())
                .with(AgentBuilder.RedefinitionStrategy.RETRANSFORMATION)
                .type(ElementMatchers.nameContains("ComputationGraph"))
                .transform(new ComputationGraphTransformer());


        agentBuilder6.installOn(inst);

        AgentBuilder agentBuilder2 =  new AgentBuilder.Default()
                .ignore(none())
                .with(AgentBuilder.RedefinitionStrategy.RETRANSFORMATION)
                .type(ElementMatchers.isSubTypeOf(Layer.class))
                .transform(new LayerTransformer());

        agentBuilder2.installOn(inst);

        AgentBuilder agentBuilder3 = new AgentBuilder.Default()
                .ignore(none())
                .with(AgentBuilder.RedefinitionStrategy.RETRANSFORMATION)
                .type(ElementMatchers.isSubTypeOf(GraphVertex.class))
                .transform(new ComputationGraphVertexTransformer());

        agentBuilder3.installOn(inst);



        AgentBuilder agentBuilder4 = new AgentBuilder.Default()
                .ignore(none())
                .with(AgentBuilder.RedefinitionStrategy.RETRANSFORMATION)
                .type(ElementMatchers.isSubTypeOf(OpExecutioner.class))
                .transform(new OpExecutionerTransformer());

        agentBuilder4.installOn(inst);


        AgentBuilder agentBuilder5 =  new AgentBuilder.Default()
                .with(AgentBuilder.RedefinitionStrategy.RETRANSFORMATION)
                .type(ElementMatchers.isSubTypeOf(INDArray.class).or(ElementMatchers.named("BaseNDArray")))
                .transform(new INDArrayTransformer());
        agentBuilder5.installOn(inst);


    }




}
