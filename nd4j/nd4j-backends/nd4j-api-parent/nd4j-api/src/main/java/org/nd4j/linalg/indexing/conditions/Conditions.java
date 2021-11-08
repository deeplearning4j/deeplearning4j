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

public class Conditions {

    private Conditions() {}

    public enum ConditionMode {
        EPSILON_EQUALS, //0
        EPSILON_NOT_EQUALS, //1
        LESS_THAN, //2
        GREATER_THAN, //3
        LESS_THAN_OR_EQUAL, //4
        GREATER_THAN_OR_EQUAL, //5
        ABS_LESS_THAN, //6
        ABS_GREATER_THAN, //7
        IS_INFINITE, //8
        IS_NAN, //9
        ABS_EQUALS, //10
        NOT_EQUALS, //11
        ABS_GREATER_OR_EQUAL, //12
        ABS_LESS_THAN_OR_EQUAL, //13
        IS_FINITE, //14
        NOT_FINITE, //15
        AGGREGATE // this is an aggregate enum for or, and, not, and other indirect conditions that depend on others

    }


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
     * This method will create Condition that checks if the absolute value of
     * x is equal to 0.0
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition absEquals() {
        // in case of pairwise MatchCondition we don't really care about number here
        return absEquals(0.0);
    }

    /**
     This method will create Condition that checks if the absolute value of
     * x is equal to the specified value.
     * PLEASE NOTE: This condition should be used only with pairwise methods, i.e. INDArray.match(...)
     * @return
     */
    public static Condition absEquals(double value) {
        // in case of pairwise MatchCondition we don't really care about number here
        return new AbsoluteEquals(value);
    }


    /**
     * This method will create Condition that checks if value is value X is less than value Y in absolute values
     *
     * @return
     */
    public static Condition absLessThan(Number value) {
        return new AbsValueLessThan(value);
    }



    public static Condition fromInt(int mode) {
        return fromInt(mode,0.0);
    }


    public static Condition fromInt(int mode,Double value) {
        if(value == null) value = 0.0;

        switch(ConditionMode.values()[mode]) {
            case IS_FINITE:
                return Conditions.isFinite();
            case IS_NAN:
                return Conditions.isInfinite();
            case LESS_THAN:
                return Conditions.lessThan(value);
            case ABS_EQUALS:
                return Conditions.absEquals(value);
            case NOT_EQUALS:
                return Conditions.notEquals();
            case NOT_FINITE:
                return Conditions.notFinite();
            case IS_INFINITE:
                return Conditions.isInfinite();
            case GREATER_THAN:
                return Conditions.greaterThan(value);
            case ABS_LESS_THAN:
                return Conditions.absLessThan(value);
            case EPSILON_EQUALS:
                return Conditions.epsEquals(value);
            case ABS_GREATER_THAN:
                return Conditions.absGreaterThan(value);
            case EPSILON_NOT_EQUALS:
                return Conditions.epsNotEquals(value);
            case LESS_THAN_OR_EQUAL:
                return Conditions.lessThanOrEqual(value);
            case ABS_GREATER_OR_EQUAL:
                return Conditions.absGreaterThan(value);
            case GREATER_THAN_OR_EQUAL:
                return Conditions.greaterThanOrEqual(value);
            case ABS_LESS_THAN_OR_EQUAL:
                return Conditions.absLessThanOrEqual(value);
            default:
                throw new IllegalArgumentException("Illegal value specified " + mode);

        }
    }


}
