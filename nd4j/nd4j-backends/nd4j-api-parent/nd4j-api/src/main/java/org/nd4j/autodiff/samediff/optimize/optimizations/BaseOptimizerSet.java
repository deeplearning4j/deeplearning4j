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

package org.nd4j.autodiff.samediff.optimize.optimizations;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Alex Black
 */
@Slf4j
public abstract class BaseOptimizerSet implements OptimizerSet {


    @Override
    public List<Optimizer> getOptimizers() {
        Method[] methods = this.getClass().getDeclaredMethods();
        List<Optimizer> out = new ArrayList<>(methods.length);
        for(Method m : methods){
            int modifiers = m.getModifiers();
            Class<?> retType = m.getReturnType();
            if(retType != null && Modifier.isPublic(modifiers) && Optimizer.class.isAssignableFrom(retType) ){
                try {
                    Optimizer o = (Optimizer) m.invoke(null);
                    out.add(o);
                } catch (IllegalAccessException | InvocationTargetException e) {
                    log.warn("Could not create optimizer from method: {}", m, e);
                }
            }
        }

        Class<?>[] declaredClasses = this.getClass().getDeclaredClasses();
        for(Class<?> c : declaredClasses){
            int modifiers = c.getModifiers();
            if(Modifier.isPublic(modifiers) && !Modifier.isAbstract(modifiers) && Optimizer.class.isAssignableFrom(c)){
                try{
                    out.add((Optimizer) c.newInstance());
                } catch (IllegalAccessException | InstantiationException e) {
                    log.warn("Could not create optimizer from inner class: {}", c, e);
                }
            }
        }

        return out;
    }
}
