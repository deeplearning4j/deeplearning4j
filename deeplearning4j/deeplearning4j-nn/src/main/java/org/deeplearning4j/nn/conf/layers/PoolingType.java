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

package org.deeplearning4j.nn.conf.layers;

/**
 * Pooling type:<br>
 * <b>MAX</b>: Max pooling - output is the maximum value of the input values<br>
 * <b>AVG</b>: Average pooling - output is the average value of the input values<br>
 * <b>SUM</b>: Sum pooling - output is the sum of the input values<br>
 * <b>PNORM</b>: P-norm pooling<br>
 */
public enum  PoolingType {
    MAX, AVG, SUM, PNORM
}
