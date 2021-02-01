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

package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.factory.Nd4j;

/**
 * Static class for conditions
 *
 * @author Adam Gibson
 */
public class Conditions {

    /**
     * This method will create Condition that checks if value is infinite
     * @return
     */
    public static Condition isInfinite() {
        return new IsInfinite();
    }

    /**
     * This method will create Condition that checks if value is NaN
     * @return
     */
    public static Condition isNan() {
        return new IsNaN();
    }

    /**
     * This method will create Condition that checks if value is finite
     * @return
     */
    public static Condition isFinite() {
        return new IsFinite();
    }

    /**
     * This method will create Condition that checks if value is NOT finite
     * @return
     */
    public static Condition notFinite() {
        return new NotFinite();
    }

    /**
     * This method will create Condition that checks if value is two values are not equal wrt eps
     *
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition epsNotEquals() {
        // in case of pairwise MatchCondition we don't really care about number here
        return epsNotEquals(0.0);
    }

    /**
     * This method will create Condition that checks if value is two values are not equal wrt eps
     *
     * @return
     */
    public static Condition epsNotEquals(Number value) {
        return new EpsilonNotEquals(value);
    }

    /**
     * This method will create Condition that checks if value is two values are equal wrt eps
     *
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition epsEquals() {
        // in case of pairwise MatchCondition we don't really care about number here
        return epsEquals(0.0);
    }

    /**
     * This method will create Condition that checks if value is two values are equal wrt eps
     *
     * @return
     */
    public static Condition epsEquals(Number value) {
        return epsEquals(value, Nd4j.EPS_THRESHOLD);
    }

    /**
     * This method will create Condition that checks if value is two values are equal wrt eps
     *
     * @return
     */
    public static Condition epsEquals(Number value, Number epsilon) {
        return new EpsilonEquals(value, epsilon.doubleValue());
    }

    /**
     * This method will create Condition that checks if value is two values are equal
     *
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition equals() {
        // in case of pairwise MatchCondition we don't really care about number here
        return equals(0.0);
    }

    /**
     * This method will create Condition that checks if value is two values are equal
     *
     * @return
     */
    public static Condition equals(Number value) {
        return new EqualsCondition(value);
    }

    /**
     * This method will create Condition that checks if value is two values are not equal
     *
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition notEquals() {
        // in case of pairwise MatchCondition we don't really care about number here
        return notEquals(0.0);
    }

    /**
     * This method will create Condition that checks if value is two values are not equal
     *
     * @return
     */
    public static Condition notEquals(Number value) {
        return new NotEqualsCondition(value);
    }

    /**
     * This method will create Condition that checks if value is value X is greater than value Y
     *
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition greaterThan() {
        // in case of pairwise MatchCondition we don't really care about number here
        return greaterThan(0.0);
    }

    /**
     * This method will create Condition that checks if value is value X is greater than value Y
     *
     * @return
     */
    public static Condition greaterThan(Number value) {
        return new GreaterThan(value);
    }

    /**
     * This method will create Condition that checks if value is value X is less than value Y
     *
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition lessThan() {
        // in case of pairwise MatchCondition we don't really care about number here
        return lessThan(0.0);
    }

    /**
     * This method will create Condition that checks if value is value X is less than value Y
     *
     * @return
     */
    public static Condition lessThan(Number value) {
        return new LessThan(value);
    }

    /**
     * This method will create Condition that checks if value is value X is less than or equal to value Y
     *
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition lessThanOrEqual() {
        // in case of pairwise MatchCondition we don't really care about number here
        return lessThanOrEqual(0.0);
    }

    /**
     * This method will create Condition that checks if value is value X is less than or equal to value Y
     *
     * @return
     */
    public static Condition lessThanOrEqual(Number value) {
        return new LessThanOrEqual(value);
    }

    /**
     * This method will create Condition that checks if value is value X is greater than or equal to value Y
     *
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition greaterThanOrEqual() {
        // in case of pairwise MatchCondition we don't really care about number here
        return greaterThanOrEqual(0.0);
    }

    /**
     * This method will create Condition that checks if value is value X is greater than or equal to value Y
     *
     * @return
     */
    public static Condition greaterThanOrEqual(Number value) {
        return new GreaterThanOrEqual(value);
    }

    /**
     * This method will create Condition that checks if value is value X is greater than or equal to value Y in absolute values
     *
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition absGreaterThanOrEqual() {
        // in case of pairwise MatchCondition we don't really care about number here
        return absGreaterThanOrEqual(0.0);
    }

    /**
     * This method will create Condition that checks if value is value X is greater than or equal to value Y in absolute values
     *
     * @return
     */
    public static Condition absGreaterThanOrEqual(Number value) {
        return new AbsValueGreaterOrEqualsThan(value);
    }

    /**
     * This method will create Condition that checks if value is value X is less than or equal to value Y in absolute values
     *
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition absLessThanOrEqual() {
        // in case of pairwise MatchCondition we don't really care about number here
        return absLessThanOrEqual(0.0);
    }

    /**
     * This method will create Condition that checks if value is value X is less than or equal to value Y in absolute values
     *
     * @return
     */
    public static Condition absLessThanOrEqual(Number value) {
        return new AbsValueLessOrEqualsThan(value);
    }

    /**
     * This method will create Condition that checks if value is value X is greater than value Y in absolute values
     *
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition absGreaterThan() {
        // in case of pairwise MatchCondition we don't really care about number here
        return absGreaterThan(0.0);
    }

    /**
     * This method will create Condition that checks if value is value X is greater than value Y in absolute values
     *
     * @return
     */
    public static Condition absGreaterThan(Number value) {
        return new AbsValueGreaterThan(value);
    }

    /**
     * This method will create Condition that checks if value is value X is less than value Y in absolute values
     *
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition absLessThan() {
        // in case of pairwise MatchCondition we don't really care about number here
        return absLessThan(0.0);
    }

    /**
     * This method will create Condition that checks if value is value X is less than value Y in absolute values
     *
     * @return
     */
    public static Condition absLessThan(Number value) {
        return new AbsValueLessThan(value);
    }

}
