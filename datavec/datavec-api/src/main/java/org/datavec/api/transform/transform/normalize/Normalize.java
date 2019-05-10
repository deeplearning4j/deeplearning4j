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

package org.datavec.api.transform.transform.normalize;

/**Built-in normalization methods.
 *
 * Normalization methods include:<br>
 * <b>MinMax</b>: (x-min)/(max-min) -> maps values to range [0,1]<br>
 * <b>MinMax2</b>: 2 * (x-min)/(max-min) + 1 -> maps values to range [-1,1]<br>
 * <b>Standardize</b>: Normalize such that output has distribution N(0,1)<br>
 * <b>SubtractMean</b>: Normalize by only subtracting the mean value<br>
 * <b>Log2Mean</b>: Normalization of the form log2((x-min)/(mean-min) + 1)<br>
 * <b>Log2MeanExcludingMin</b>: As per Log2Mean, but the 'mean' is calculated excluding the minimum value.<br>
 *
 *
 * @author Alex Black
 */
public enum Normalize {
    MinMax, MinMax2, Standardize, SubtractMean, Log2Mean, Log2MeanExcludingMin

}
