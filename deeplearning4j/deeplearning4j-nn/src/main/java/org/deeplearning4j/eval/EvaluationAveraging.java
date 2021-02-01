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

package org.deeplearning4j.eval;

/**
 * @deprecated Use {@link org.nd4j.evaluation.EvaluationAveraging}
 */
@Deprecated
public enum EvaluationAveraging {
    Macro, Micro;

    public org.nd4j.evaluation.EvaluationAveraging toNd4j(){
        switch (this){
            case Macro:
                return org.nd4j.evaluation.EvaluationAveraging.Macro;
            case Micro:
                return org.nd4j.evaluation.EvaluationAveraging.Micro;
        }
        throw new UnsupportedOperationException("Unknown: " + this);
    }
}
