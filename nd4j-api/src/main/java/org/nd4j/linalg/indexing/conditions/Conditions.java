/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.indexing.conditions;

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
        return new EqualsCondition(value);
    }

    public static Condition epsEquals(Number value) {
        return new EqualsCondition(value);
    }

    public static Condition equals(IComplexNumber value) {
        return new EqualsCondition(value);
    }

    public static Condition equals(Number value) {
        return new EqualsCondition(value);
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

    public static Condition greaterThanOEqual(Number value) {
        return new GreaterThanOrEqual(value);
    }

}
