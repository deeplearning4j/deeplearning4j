/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.ops.factory;

import org.nd4j.linalg.ops.factory.impl.*;

/**
 * @author Adam Gibson
 */
public class ElementWiseOpFactories {

    private static ElementWiseOpFactory ABS = new AbsElementWiseOpFactory();
    private static ElementWiseOpFactory EQUAL_TO = new EqualToElementWiseOpFactory();
    private static ElementWiseOpFactory NOT_EQUAL_TO = new NotEqualToElementWiseOpFactory();
    private static ElementWiseOpFactory EXP = new ExpElementWiseOpFactory();
    private static ElementWiseOpFactory FLOOR = new FloorElementWiseOpFactory();
    private static ElementWiseOpFactory LOG = new LogElementWiseOpFactory();
    private static ElementWiseOpFactory ROUND = new RoundElementWiseOpFactory();
    private static ElementWiseOpFactory SIGN = new SignElementWiseOpFactory();
    private static ElementWiseOpFactory TANH = new TanhElementWiseOpFactory();
    private static ElementWiseOpFactory HARD_TANH = new HardTanhElementWiseOpFactory();
    private static ElementWiseOpFactory IDENTITY = new IdentityElementWiseOpFactory();
    private static ElementWiseOpFactory NEGATIVE = new NegativeElementWiseOpFactory();
    private static ElementWiseOpFactory SQRT = new SqrtElementWiseOpFactory();
    private static ElementWiseOpFactory SIGMOID = new SigmoidElementWiseOpFactory();
    private static ElementWiseOpFactory LESS_THAN = new LessThanElementWiseOpFactory();
    private static ElementWiseOpFactory MAX_OUT = new MaxOutElementWiseOpFactory();
    private static ElementWiseOpFactory GREATER_THAN = new GreaterThanElementWiseOpFactory();


    public static ElementWiseOpFactory greaterThan() {
        return GREATER_THAN;
    }

    public static ElementWiseOpFactory hardTanh() {
        return HARD_TANH;
    }

    public static ElementWiseOpFactory sigmoid() {
        return SIGMOID;
    }

    public static ElementWiseOpFactory abs() {
        return ABS;
    }

    public static ElementWiseOpFactory equalTo() {
        return EQUAL_TO;
    }

    public static ElementWiseOpFactory notEqualTo() {
        return NOT_EQUAL_TO;
    }

    public static ElementWiseOpFactory exp() {
        return EXP;
    }

    public static ElementWiseOpFactory floor() {
        return FLOOR;
    }

    public static ElementWiseOpFactory log() {
        return LOG;
    }

    public static ElementWiseOpFactory round() {
        return ROUND;
    }

    public static ElementWiseOpFactory sign() {
        return SIGN;
    }

    public static ElementWiseOpFactory tanh() {
        return TANH;
    }

    public static ElementWiseOpFactory identity() {
        return IDENTITY;
    }

    public static ElementWiseOpFactory negative() {
        return NEGATIVE;
    }

    public static ElementWiseOpFactory sqrt() {
        return SQRT;
    }

    public static ElementWiseOpFactory maxOut() {
        return MAX_OUT;
    }

    public static ElementWiseOpFactory max() {
        return new MaxElementWiseOpFactory();
    }

    public static ElementWiseOpFactory min() {
        return new MinElementWiseOpFactory();
    }


    public static ElementWiseOpFactory stabilize() {
        return new StabilizeElementWiseOpFactory();
    }

    public static ElementWiseOpFactory pow() {
        return new PowElementWiseOpFactory();
    }

    public static ElementWiseOpFactory lessThan() {
        return LESS_THAN;
    }


}
