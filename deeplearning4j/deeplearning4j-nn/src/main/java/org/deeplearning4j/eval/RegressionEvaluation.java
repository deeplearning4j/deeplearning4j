/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

import lombok.Data;
import lombok.EqualsAndHashCode;

import java.util.List;

/**
 * @deprecated Use ND4J RegressionEvaluation class, which has the same interface: {@link org.nd4j.evaluation.regression.RegressionEvaluation}
 */
@Deprecated
@Data
@EqualsAndHashCode(callSuper = true)
public class RegressionEvaluation extends org.nd4j.evaluation.regression.RegressionEvaluation implements org.deeplearning4j.eval.IEvaluation<org.nd4j.evaluation.regression.RegressionEvaluation> {

    /**
     * @deprecated Use ND4J RegressionEvaluation class, which has the same interface: {@link org.nd4j.evaluation.regression.RegressionEvaluation.Metric}
     */
    @Deprecated
    public enum Metric { MSE, MAE, RMSE, RSE, PC, R2;
        public boolean minimize(){
            return toNd4j().minimize();
        }

        public org.nd4j.evaluation.regression.RegressionEvaluation.Metric toNd4j(){
            switch (this){
                case MSE:
                    return org.nd4j.evaluation.regression.RegressionEvaluation.Metric.MSE;
                case MAE:
                    return org.nd4j.evaluation.regression.RegressionEvaluation.Metric.MAE;
                case RMSE:
                    return org.nd4j.evaluation.regression.RegressionEvaluation.Metric.RMSE;
                case RSE:
                    return org.nd4j.evaluation.regression.RegressionEvaluation.Metric.RSE;
                case PC:
                    return org.nd4j.evaluation.regression.RegressionEvaluation.Metric.PC;
                case R2:
                    return org.nd4j.evaluation.regression.RegressionEvaluation.Metric.R2;
                default:
                    throw new IllegalStateException("Unknown enum: " + this);
            }
        }
    }

    /**
     * @deprecated Use ND4J RegressionEvaluation class, which has the same interface: {@link org.nd4j.evaluation.regression.RegressionEvaluation}
     */
    @Deprecated
    public RegressionEvaluation() { }

    /**
     * @deprecated Use ND4J RegressionEvaluation class, which has the same interface: {@link org.nd4j.evaluation.regression.RegressionEvaluation}
     */
    @Deprecated
    public RegressionEvaluation(long nColumns) {
        super(nColumns);
    }

    /**
     * @deprecated Use ND4J RegressionEvaluation class, which has the same interface: {@link org.nd4j.evaluation.regression.RegressionEvaluation}
     */
    @Deprecated
    public RegressionEvaluation(long nColumns, long precision) {
        super(nColumns, precision);
    }

    /**
     * @deprecated Use ND4J RegressionEvaluation class, which has the same interface: {@link org.nd4j.evaluation.regression.RegressionEvaluation}
     */
    @Deprecated
    public RegressionEvaluation(String... columnNames) {
        super(columnNames);
    }

    /**
     * @deprecated Use ND4J RegressionEvaluation class, which has the same interface: {@link org.nd4j.evaluation.regression.RegressionEvaluation}
     */
    @Deprecated
    public RegressionEvaluation(List<String> columnNames) {
        super(columnNames);
    }

    /**
     * @deprecated Use ND4J RegressionEvaluation class, which has the same interface: {@link org.nd4j.evaluation.regression.RegressionEvaluation}
     */
    @Deprecated
    public RegressionEvaluation(List<String> columnNames, long precision) {
        super(columnNames, precision);
    }

    /**
     * @deprecated Use ND4J RegressionEvaluation class, which has the same interface: {@link org.nd4j.evaluation.regression.RegressionEvaluation}
     */
    @Deprecated
    public double scoreForMetric(Metric metric){
        return scoreForMetric(metric.toNd4j());
    }
}
