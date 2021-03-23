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
package org.datavec.api.util;

import org.datavec.api.timeseries.util.TimeSeriesWritableUtils;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.ArrayList;
import java.util.List;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("Time Series Utils Test")
@Tag(TagNames.JAVA_ONLY)
@Tag(TagNames.FILE_IO)
class TimeSeriesUtilsTest extends BaseND4JTest {

    @Test
    @DisplayName("Test Time Series Creation")
    void testTimeSeriesCreation() {
        List<List<List<Writable>>> test = new ArrayList<>();
        List<List<Writable>> timeStep = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            timeStep.add(getRecord(5));
        }
        test.add(timeStep);
        INDArray arr = TimeSeriesWritableUtils.convertWritablesSequence(test).getFirst();
        assertArrayEquals(new long[] { 1, 5, 5 }, arr.shape());
    }

    private List<Writable> getRecord(int length) {
        List<Writable> ret = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            ret.add(new DoubleWritable(1.0));
        }
        return ret;
    }
}
