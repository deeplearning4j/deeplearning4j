/*
 * Copyright (c) 2015-2019 Skymind, Inc.
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
 */

package org.nd4j.evaluation;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.nd4j.evaluation.custom.CustomEvaluation;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.primitives.Pair;

public class CustomEvaluationTest extends BaseNd4jTest {

    public CustomEvaluationTest(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void customEvalTest(){
        CustomEvaluation accuracyEval = new CustomEvaluation<Pair<Number, Long>>(
                (labels, pred, mask, meta) -> new Pair<>(labels.eq(pred).castTo(DataType.INT).sumNumber(), labels.size(0)),
                CustomEvaluation.mergeConcatenate());

        accuracyEval.eval(Nd4j.createFromArray(1, 1, 2, 1, 3), Nd4j.createFromArray(1, 1, 4, 1, 2));

        double acc = accuracyEval.getValue(new CustomEvaluation.Metric<Pair<Number, Long>>(
                (list) -> {
                    int sum = 0;
                    int count = 0;
                    for(Pair<Number, Long> p : list){
                        sum += p.getFirst().intValue();
                        count += p.getSecond();
                    }
                    return ((double) (sum)) / count;
                }
        ));

        assertEquals("Accuracy", acc, 3.0/5, 0.001);

    }

}
