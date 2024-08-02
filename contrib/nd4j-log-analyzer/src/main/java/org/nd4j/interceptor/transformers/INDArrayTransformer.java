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
import org.nd4j.interceptor.advice.INDArrayCreationAdvice;
import org.nd4j.interceptor.advice.INDArrayUpdateAdvice;
import org.nd4j.linalg.api.buffer.DataBuffer;

import java.security.ProtectionDomain;

public class INDArrayTransformer implements AgentBuilder.Transformer {
    public DynamicType.Builder<?> transform(DynamicType.Builder<?> builder, TypeDescription typeDescription, ClassLoader classLoader, JavaModule javaModule) {
        builder = builder.visit(Advice.to(INDArrayCreationAdvice.class).on(ElementMatchers.isConstructor()));

        builder = builder.visit(Advice.to(INDArrayUpdateAdvice.class).on(ElementMatchers.named("assign")));
        builder = builder.visit(Advice.to(INDArrayUpdateAdvice.class).on(ElementMatchers.named("putScalar")));
        builder = builder.visit(Advice.to(INDArrayUpdateAdvice.class).on(ElementMatchers.named("put")));

        return builder;
    }

    @Override
    public DynamicType.Builder<?> transform(DynamicType.Builder<?> builder, TypeDescription typeDescription, ClassLoader classLoader, JavaModule javaModule, ProtectionDomain protectionDomain) {
        return transform(builder, typeDescription, classLoader, javaModule);
    }
}
