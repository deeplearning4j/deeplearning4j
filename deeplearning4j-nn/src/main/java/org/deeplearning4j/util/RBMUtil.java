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
 */

package org.deeplearning4j.util;


import org.deeplearning4j.nn.conf.layers.RBM;

/**
 * Handles various cooccurrences for RBM specific cooccurrences
 * @author Adam Gibson
 */
public class RBMUtil {

    private RBMUtil() {}

    public static RBM.VisibleUnit inverse(RBM.HiddenUnit visible) {
        switch(visible) {
            case BINARY:
                return  RBM.VisibleUnit.BINARY;
            case GAUSSIAN:
                return  RBM.VisibleUnit.GAUSSIAN;
            case SOFTMAX:
                return  RBM.VisibleUnit.SOFTMAX;
            default:
                return null;

        }
    }

    public static RBM.HiddenUnit inverse( RBM.VisibleUnit hidden) {
        switch(hidden) {
            case BINARY:
                return   RBM.HiddenUnit.BINARY;
            case GAUSSIAN:
                return  RBM.HiddenUnit.GAUSSIAN;
            case SOFTMAX:
                return  RBM.HiddenUnit.SOFTMAX;
        }

        return null;
    }


}
