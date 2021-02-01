/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.adapters;

/**
 * This interface describes methods needed to convert custom JVM objects to INDArrays, suitable for feeding neural networks
 *
 * @param <I> type of the Input for the model. I.e. String for raw text
 * @param <O> type of the Output for the model, I.e. Sentiment, for Text->Sentiment extraction
 *
 * @author raver119@gmail.com
 */
public interface InferenceAdapter<I, O> extends InputAdapter<I>, OutputAdapter<O> {
}
