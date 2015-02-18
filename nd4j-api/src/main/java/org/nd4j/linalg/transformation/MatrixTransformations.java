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

package org.nd4j.linalg.transformation;

public class MatrixTransformations {

    public static MatrixTransform multiplyScalar(double num) {
        return new MultiplyScalar(num);
    }

    public static MatrixTransform addScalar(double num) {
        return new AddScalar(num);
    }

    public static MatrixTransform divideScalar(double num) {
        return new DivideScalar(num);
    }


    public static MatrixTransform sqrt() {
        return new SqrtScalar();
    }

    public static MatrixTransform exp() {
        return new ExpTransform();
    }

    public static MatrixTransform log() {
        return new LogTransform();
    }


    public static MatrixTransform powScalar(double num) {
        return new PowScale(num);
    }

}
