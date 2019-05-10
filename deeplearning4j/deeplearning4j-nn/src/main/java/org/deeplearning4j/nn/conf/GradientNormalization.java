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

package org.deeplearning4j.nn.conf;

/**Gradient normalization strategies. These are applied on raw gradients, before the gradients are passed to the
 * updater (SGD, RMSProp, Momentum, etc)<br>
 * <p><b>None</b> = no gradient normalization (default)<br></p>
 *
 * <p><b>RenormalizeL2PerLayer</b> = rescale gradients by dividing by the L2 norm of all gradients for the layer.<br></p>
 *
 * <p><b>RenormalizeL2PerParamType</b> = rescale gradients by dividing by the L2 norm of the gradients, separately for
 * each type of parameter within the layer.<br>
 * This differs from RenormalizeL2PerLayer in that here, each parameter type (weight, bias etc) is normalized separately.<br>
 * For example, in a MLP/FeedForward network (where G is the gradient vector), the output is as follows:
 * <ul style="list-style-type:none">
 *     <li>GOut_weight = G_weight / l2(G_weight)</li>
 *     <li>GOut_bias = G_bias / l2(G_bias)</li>
 * </ul>
 * </p>
 *
 * <p><b>ClipElementWiseAbsoluteValue</b> = clip the gradients on a per-element basis.<br>
 * For each gradient g, set g <- sign(g)*max(maxAllowedValue,|g|).<br>
 * i.e., if a parameter gradient has absolute value greater than the threshold, truncate it.<br>
 * For example, if threshold = 5, then values in range -5&lt;g&lt;5 are unmodified; values &lt;-5 are set
 * to -5; values &gt;5 are set to 5.<br>
 * This was proposed by Mikolov (2012), <i>Statistical Language Models Based on Neural Networks</i> (thesis),
 * <a href="http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf">http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf</a>
 * in the context of learning recurrent neural networks.<br>
 * Threshold for clipping can be set in Layer configuration, using gradientNormalizationThreshold(double threshold)
 * </p>
 *
 * <p><b>ClipL2PerLayer</b> = conditional renormalization. Somewhat similar to RenormalizeL2PerLayer, this strategy
 * scales the gradients <i>if and only if</i> the L2 norm of the gradients (for entire layer) exceeds a specified
 * threshold. Specifically, if G is gradient vector for the layer, then:
 * <ul style="list-style-type:none">
 *     <li>GOut = G &nbsp;&nbsp;&nbsp; if l2Norm(G) < threshold (i.e., no change) </li>
 *     <li>GOut = threshold * G / l2Norm(G) &nbsp;&nbsp;&nbsp; otherwise </li>
 * </ul>
 * Thus, the l2 norm of the scaled gradients will not exceed the specified threshold, though may be smaller than it<br>
 * See: Pascanu, Mikolov, Bengio (2012), <i>On the difficulty of training Recurrent Neural Networks</i>,
 * <a href="http://arxiv.org/abs/1211.5063">http://arxiv.org/abs/1211.5063</a><br>
 * Threshold for clipping can be set in Layer configuration, using gradientNormalizationThreshold(double threshold)
 * </p>
 *
 * <p><b>ClipL2PerParamType</b> = conditional renormalization. Very similar to ClipL2PerLayer, however instead of clipping
 * per layer, do clipping on each parameter type separately.<br>
 * For example in a recurrent neural network, input weight gradients, recurrent weight gradients and bias gradient are all
 * clipped separately. Thus if one set of gradients are very large, these may be clipped while leaving the other gradients
 * unmodified.<br>
 * Threshold for clipping can be set in Layer configuration, using gradientNormalizationThreshold(double threshold)</p>
 *
 * @author Alex Black
 */
public enum GradientNormalization {
    None, RenormalizeL2PerLayer, RenormalizeL2PerParamType, ClipElementWiseAbsoluteValue, ClipL2PerLayer, ClipL2PerParamType
}
