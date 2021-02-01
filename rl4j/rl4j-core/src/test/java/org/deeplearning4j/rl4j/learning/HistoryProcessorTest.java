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

package org.deeplearning4j.rl4j.learning;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 *
 * @author saudet
 */
public class HistoryProcessorTest {

    @Test
    public void testHistoryProcessor() {
        HistoryProcessor.Configuration conf = HistoryProcessor.Configuration.builder()
                .croppingHeight(2).croppingWidth(2).rescaledHeight(2).rescaledWidth(2).build();
        IHistoryProcessor hp = new HistoryProcessor(conf);
        INDArray a = Nd4j.createFromArray(new float[][][] {{{0.1f, 0.1f, 0.1f}, {0.2f, 0.2f, 0.2f}}, {{0.3f, 0.3f, 0.3f}, {0.4f, 0.4f, 0.4f}}});
        hp.add(a);
        hp.add(a);
        hp.add(a);
        hp.add(a);
        INDArray[] h = hp.getHistory();
        assertEquals(4, h.length);
        assertEquals(           1, h[0].shape()[0]);
        assertEquals(a.shape()[0], h[0].shape()[1]);
        assertEquals(a.shape()[1], h[0].shape()[2]);
        assertEquals(0.1f * hp.getScale(), h[0].getDouble(0, 0, 0), 1);
        assertEquals(0.2f * hp.getScale(), h[0].getDouble(0, 0, 1), 1);
        assertEquals(0.3f * hp.getScale(), h[0].getDouble(0, 1, 0), 1);
        assertEquals(0.4f * hp.getScale(), h[0].getDouble(0, 1, 1), 1);
    }
}
