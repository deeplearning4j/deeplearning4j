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

package org.deeplearning4j.eval.curves;

import lombok.Getter;
import lombok.NonNull;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Created by Alex on 05/07/2017.
 */
@Getter
public class ReliabilityDiagram extends BaseCurve {

    private final String title;
    private final double[] meanPredictedValueX;
    private final double[] fractionPositivesY;


    public ReliabilityDiagram(@JsonProperty("title") String title,
                    @NonNull @JsonProperty("meanPredictedValueX") double[] meanPredictedValueX,
                    @NonNull @JsonProperty("fractionPositivesY") double[] fractionPositivesY) {
        this.title = title;
        this.meanPredictedValueX = meanPredictedValueX;
        this.fractionPositivesY = fractionPositivesY;
    }

    @Override
    public int numPoints() {
        return meanPredictedValueX.length;
    }

    @Override
    public double[] getX() {
        return getMeanPredictedValueX();
    }

    @Override
    public double[] getY() {
        return getFractionPositivesY();
    }

    @Override
    public String getTitle() {
        if (title == null) {
            return "Reliability Diagram";
        }
        return title;
    }
}
