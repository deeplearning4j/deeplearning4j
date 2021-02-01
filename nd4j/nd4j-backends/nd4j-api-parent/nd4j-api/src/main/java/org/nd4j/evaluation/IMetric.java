/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.evaluation;

/**
 * A metric used to get a double value from an {@link IEvaluation}.
 *
 * Examples: {@link org.nd4j.evaluation.classification.Evaluation.Metric#ACCURACY}, {@link org.nd4j.evaluation.classification.ROC.Metric#AUPRC}.
 */
public interface IMetric {

    /**
     * The {@link IEvaluation} class this metric is for
     */
    public Class<? extends IEvaluation> getEvaluationClass();

    /**
     * Whether this metric should be minimized (aka whether lower values are better).
     */
    public boolean minimize();
}
