/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff.transform;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * An OpPredicate defines whether an operation ({@link DifferentialFunction}) matches or not.<br>
 *
 * @author Alex Black
 */
public abstract class OpPredicate {

    /**
     *
     * @param sameDiff SameDiff instance the function belongs to
     * @param function
     * @return Returns whether the specific function matches the predicate
     */
    public abstract boolean matches(SameDiff sameDiff, DifferentialFunction function);


    /**
     * Return true if the operation own (user specified) name equals the specified name
     */
    public static OpPredicate nameEquals(final String name){
        return new OpPredicate() {
            @Override
            public boolean matches(SameDiff sameDiff, DifferentialFunction function) {
                return function.getOwnName().equals(name);
            }
        };
    }

    /**
     * Return true if the operation name (i.e., "add", "mul", etc - not the user specified name) equals the specified name
     */
    public static OpPredicate opNameEquals(final String opName){
        return new OpPredicate() {
            @Override
            public boolean matches(SameDiff sameDiff, DifferentialFunction function) {
                return function.opName().equals(opName);
            }
        };
    }

    /**
     * Return true if the operation own (user specified) name matches the specified regular expression
     */
    public static OpPredicate nameMatches(final String regex){
        return new OpPredicate() {
            @Override
            public boolean matches(SameDiff sameDiff, DifferentialFunction function) {
                return function.getOwnName().matches(regex);
            }
        };
    }

    /**
     * Return true if the operation name (i.e., "add", "mul", etc - not the user specified name) matches the specified regular expression
     */
    public static OpPredicate opNameMatches(final String regex){
        return new OpPredicate() {
            @Override
            public boolean matches(SameDiff sameDiff, DifferentialFunction function) {
                return function.getOwnName().matches(regex);
            }
        };
    }

    /**
     * Return true if the operation class is equal to the specified class
     */
    public static OpPredicate classEquals(final Class<?> c){
        return new OpPredicate() {
            @Override
            public boolean matches(SameDiff sameDiff, DifferentialFunction function) {
                return function.getClass() == c;
            }
        };
    }


}
