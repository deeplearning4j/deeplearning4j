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

package org.datavec.api.transform.condition;

import java.util.Set;

/**
 * Created by Alex on 24/03/2016.
 */
public enum ConditionOp {
    LessThan, LessOrEqual, GreaterThan, GreaterOrEqual, Equal, NotEqual, InSet, NotInSet;


    public boolean apply(double x, double value, Set<Double> set) {
        switch (this) {
            case LessThan:
                return x < value;
            case LessOrEqual:
                return x <= value;
            case GreaterThan:
                return x > value;
            case GreaterOrEqual:
                return x >= value;
            case Equal:
                return x == value;
            case NotEqual:
                return x != value;
            case InSet:
                return set.contains(x);
            case NotInSet:
                return !set.contains(x);
            default:
                throw new RuntimeException("Unknown or not implemented op: " + this);
        }
    }


    public boolean apply(float x, float value, Set<Float> set) {
        switch (this) {
            case LessThan:
                return x < value;
            case LessOrEqual:
                return x <= value;
            case GreaterThan:
                return x > value;
            case GreaterOrEqual:
                return x >= value;
            case Equal:
                return x == value;
            case NotEqual:
                return x != value;
            case InSet:
                return set.contains(x);
            case NotInSet:
                return !set.contains(x);
            default:
                throw new RuntimeException("Unknown or not implemented op: " + this);
        }
    }

    public boolean apply(int x, int value, Set<Integer> set) {
        switch (this) {
            case LessThan:
                return x < value;
            case LessOrEqual:
                return x <= value;
            case GreaterThan:
                return x > value;
            case GreaterOrEqual:
                return x >= value;
            case Equal:
                return x == value;
            case NotEqual:
                return x != value;
            case InSet:
                return set.contains(x);
            case NotInSet:
                return !set.contains(x);
            default:
                throw new RuntimeException("Unknown or not implemented op: " + this);
        }
    }

    public boolean apply(long x, long value, Set<Long> set) {
        switch (this) {
            case LessThan:
                return x < value;
            case LessOrEqual:
                return x <= value;
            case GreaterThan:
                return x > value;
            case GreaterOrEqual:
                return x >= value;
            case Equal:
                return x == value;
            case NotEqual:
                return x != value;
            case InSet:
                return set.contains(x);
            case NotInSet:
                return !set.contains(x);
            default:
                throw new RuntimeException("Unknown or not implemented op: " + this);
        }
    }


    public boolean apply(String x, String value, Set<String> set) {
        switch (this) {
            case Equal:
                return value.equals(x);
            case NotEqual:
                return !value.equals(x);
            case InSet:
                return set.contains(x);
            case NotInSet:
                return !set.contains(x);
            case LessThan:
            case LessOrEqual:
            case GreaterThan:
            case GreaterOrEqual:
                throw new UnsupportedOperationException("Cannot use ConditionOp \"" + this + "\" on Strings");
            default:
                throw new RuntimeException("Unknown or not implemented op: " + this);
        }
    }



}
