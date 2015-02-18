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

package org.nd4j.linalg.indexing;

import com.google.common.base.Function;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 10/1/14.
 */
public class Apply {

    private INDArray toTransform;
    private Function apply;

    public void apply() {
        if (toTransform instanceof IComplexNDArray) {
            IComplexNDArray linear = (IComplexNDArray) toTransform.linearView();
            for (int i = 0; i < linear.length(); i++) {

            }
        } else {
            INDArray linear = toTransform.linearView();
            for (int i = 0; i < linear.length(); i++) {

            }
        }

    }

}
