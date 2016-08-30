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

package org.deeplearning4j.nn.weights;

/**Weight initialization scheme
 *
 * Distribution: Sample weights from a distribution based on shape of input
 * Normalized: Normalize sample weights
 * Size: Sample weights from bound uniform distribution using shape for min and max
 * Uniform: Sample weights from bound uniform distribution (specify min and max)
 * VI: Sample weights from variance normalized initialization (Glorot)
 * Zeros: Generate weights as zeros
 * Xavier:
 * RELU: N(0,2/nIn): He et al. (2015), Delving Deep into Rectifiers
 * @author Adam Gibson
 */
public enum WeightInit {
    /*
        TBD: Sparse initialization (SI) (Martens)
     */
    DISTRIBUTION,NORMALIZED,SIZE,UNIFORM,VI,ZERO,XAVIER,RELU

}
