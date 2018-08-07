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

package org.deeplearning4j.nn.simple.multiclass;

import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeNotNull;

/**
 * Created by agibsonccc on 4/28/17.
 */
public class RankClassificationResultTest {
    @Test
    public void testOutcome() {
        RankClassificationResult result =
                        new RankClassificationResult(Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2));
        assumeNotNull(result.getLabels());
        assertEquals("1", result.maxOutcomeForRow(0));
        assertEquals("1", result.maxOutcomeForRow(1));
        List<String> maxOutcomes = result.maxOutcomes();
        assertEquals(2, result.maxOutcomes().size());
        for (int i = 0; i < 2; i++) {
            assertEquals("1", maxOutcomes.get(i));
        }
    }


}
