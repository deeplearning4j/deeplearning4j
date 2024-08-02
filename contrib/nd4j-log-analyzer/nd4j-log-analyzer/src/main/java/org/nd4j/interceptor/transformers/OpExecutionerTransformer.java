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

package org.nd4j.interceptor.transformers;

import net.bytebuddy.agent.builder.AgentBuilder;
import net.bytebuddy.asm.Advice;
import net.bytebuddy.description.type.TypeDescription;
import net.bytebuddy.dynamic.DynamicType;
import net.bytebuddy.matcher.ElementMatchers;
import net.bytebuddy.utility.JavaModule;
import org.nd4j.interceptor.advice.CustomOpAdvice;
import org.nd4j.interceptor.advice.OpExecutionerAdvice;
import org.nd4j.linalg.api.ops.*;

import java.security.ProtectionDomain;
import java.util.Random;

public class OpExecutionerTransformer implements AgentBuilder.Transformer {

    public DynamicType.Builder<?> transform(DynamicType.Builder<?> builder, TypeDescription typeDescription, ClassLoader classLoader, JavaModule javaModule) {
        builder = builder.visit(Advice.to(OpExecutionerAdvice.class).on(ElementMatchers.named("execAndReturn").and(ElementMatchers.takesArguments(1)).and(ElementMatchers.takesArgument(0, TransformOp.class))));
        builder = builder.visit(Advice.to(OpExecutionerAdvice.class).on(ElementMatchers.named("execAndReturn").and(ElementMatchers.takesArguments(1)).and(ElementMatchers.takesArgument(0, ReduceOp.class))));
        builder = builder.visit(Advice.to(OpExecutionerAdvice.class).on(ElementMatchers.named("execAndReturn").and(ElementMatchers.takesArguments(1)).and(ElementMatchers.takesArgument(0, IndexAccumulation.class))));
        builder = builder.visit(Advice.to(OpExecutionerAdvice.class).on(ElementMatchers.named("execAndReturn").and(ElementMatchers.takesArguments(1)).and(ElementMatchers.takesArgument(0, ScalarOp.class))));
        builder = builder.visit(Advice.to(OpExecutionerAdvice.class).on(ElementMatchers.named("execAndReturn").and(ElementMatchers.takesArguments(1)).and(ElementMatchers.takesArgument(0, BroadcastOp.class))));
        builder = builder.visit(Advice.to(OpExecutionerAdvice.class).on(ElementMatchers.named("exec").and(ElementMatchers.takesArguments(1)).and(ElementMatchers.takesArgument(0, ReduceOp.class))));
        builder = builder.visit(Advice.to(OpExecutionerAdvice.class).on(ElementMatchers.named("exec").and(ElementMatchers.takesArguments(1)).and(ElementMatchers.takesArgument(0, BroadcastOp.class))));
        builder = builder.visit(Advice.to(OpExecutionerAdvice.class).on(ElementMatchers.named("exec").and(ElementMatchers.takesArguments(1)).and(ElementMatchers.takesArgument(0, ScalarOp.class))));
        builder = builder.visit(Advice.to(OpExecutionerAdvice.class).on(ElementMatchers.named("exec").and(ElementMatchers.takesArguments(1)).and(ElementMatchers.takesArgument(0, IndexAccumulation.class))));
        builder = builder.visit(Advice.to(OpExecutionerAdvice.class).on(ElementMatchers.named("exec").and(ElementMatchers.takesArguments(1)).and(ElementMatchers.takesArgument(0, MetaOp.class))));
        builder = builder.visit(Advice.to(OpExecutionerAdvice.class).on(ElementMatchers.named("exec").and(ElementMatchers.takesArguments(1)).and(ElementMatchers.takesArgument(0, GridOp.class))));
        builder = builder.visit(Advice.to(OpExecutionerAdvice.class).on(ElementMatchers.named("exec").and(ElementMatchers.takesArguments(1)).and(ElementMatchers.takesArgument(0, RandomOp.class))));
        builder = builder.visit(Advice.to(OpExecutionerAdvice.class).on(ElementMatchers.named("exec").and(ElementMatchers.takesArguments(2)).and(ElementMatchers.takesArgument(0, RandomOp.class)).and(ElementMatchers.takesArgument(1, Random.class))));
        builder = builder.visit(Advice.to(CustomOpAdvice.class).on(ElementMatchers.named("exec").and(ElementMatchers.takesArguments(2)).and(ElementMatchers.takesArgument(0, CustomOp.class)).and(ElementMatchers.takesArgument(1, OpContext.class))));
        return builder;

    }

    public DynamicType.Builder<?> transform(DynamicType.Builder<?> builder, TypeDescription typeDescription, ClassLoader classLoader, JavaModule javaModule, ProtectionDomain protectionDomain) {
        return transform(builder, typeDescription, classLoader, javaModule);
    }
}
