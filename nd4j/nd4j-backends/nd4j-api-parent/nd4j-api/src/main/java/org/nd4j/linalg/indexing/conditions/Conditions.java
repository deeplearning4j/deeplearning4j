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

package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Static class for conditions
 *
 * @author Adam Gibson
 */
public class Conditions {


    public static Condition isInfinite() {
        return new IsInfinite();
    }


    public static Condition isNan() {
        return new IsNaN();
    }

    public static Condition epsEquals(IComplexNumber value) {
        return new EpsilonEquals(value);
    }

    public static Condition epsNotEquals(Number value) {
        return new EpsilonNotEquals(value);
    }

    public static Condition epsEquals(Number value) {
        return epsEquals(value, Nd4j.EPS_THRESHOLD);
    }

    public static Condition epsEquals(Number value, Number epsilon) {
        return new EpsilonEquals(value, epsilon.doubleValue());
    }



    public static Condition equals(IComplexNumber value) {
        return new EqualsCondition(value);
    }

    public static Condition equals(Number value) {
        return new EqualsCondition(value);
    }

    public static Condition notEquals(Number value) {
        return new NotEqualsCondition(value);
    }

    public static Condition greaterThan(IComplexNumber value) {
        return new GreaterThan(value);
    }

    public static Condition greaterThan(Number value) {
        return new GreaterThan(value);
    }

    public static Condition lessThan(IComplexNumber value) {
        return new GreaterThan(value);
    }

    public static Condition lessThan(Number value) {
        return new LessThan(value);
    }

    public static Condition lessThanOrEqual(IComplexNumber value) {
        return new LessThanOrEqual(value);
    }

    public static Condition lessThanOrEqual(Number value) {
        return new LessThanOrEqual(value);
    }

    public static Condition greaterThanOrEqual(IComplexNumber value) {
        return new GreaterThanOrEqual(value);
    }

    public static Condition greaterThanOrEqual(Number value) {
        return new GreaterThanOrEqual(value);
    }

    public static Condition absGreaterThanOrEqual(Number value) {
        return new AbsValueGreaterOrEqualsThan(value);
    }

    public static Condition absLessThanOrEqual(Number value) {
        return new AbsValueLessOrEqualsThan(value);
    }

    public static Condition absGreaterThan(Number value) {
        return new AbsValueGreaterThan(value);
    }

    public static Condition absLessThan(Number value) {
        return new AbsValueLessThan(value);
    }

}
