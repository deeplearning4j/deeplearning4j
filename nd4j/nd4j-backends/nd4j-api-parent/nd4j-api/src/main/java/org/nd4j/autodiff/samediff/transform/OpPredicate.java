/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.autodiff.samediff.transform;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

public abstract class OpPredicate {

    public abstract boolean matches(SameDiff sameDiff, DifferentialFunction function);


    public static OpPredicate nameEquals(final String name){
        return new OpPredicate() {
            @Override
            public boolean matches(SameDiff sameDiff, DifferentialFunction function) {
                return function.getOwnName().equals(name);
            }
        };
    }

    public static OpPredicate opNameEquals(final String opName){
        return new OpPredicate() {
            @Override
            public boolean matches(SameDiff sameDiff, DifferentialFunction function) {
                return function.opName().equals(opName);
            }
        };
    }

    public static OpPredicate nameMatches(final String regex){
        return new OpPredicate() {
            @Override
            public boolean matches(SameDiff sameDiff, DifferentialFunction function) {
                return function.getOwnName().matches(regex);
            }
        };
    }

    public static OpPredicate opNameMatches(final String regex){
        return new OpPredicate() {
            @Override
            public boolean matches(SameDiff sameDiff, DifferentialFunction function) {
                return function.getOwnName().matches(regex);
            }
        };
    }

    public static OpPredicate classEquals(final Class<?> c){
        return new OpPredicate() {
            @Override
            public boolean matches(SameDiff sameDiff, DifferentialFunction function) {
                return function.getClass() == c;
            }
        };
    }


}
