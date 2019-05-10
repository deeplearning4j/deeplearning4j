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

package org.datavec.spark.transform;

import org.datavec.spark.transform.model.SingleCSVRecord;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by agibsonccc on 2/12/17.
 */
public class SingleCSVRecordTest {

    @Test(expected = IllegalArgumentException.class)
    public void testVectorAssertion() {
        DataSet dataSet = new DataSet(Nd4j.create(2, 2), Nd4j.create(1, 1));
        SingleCSVRecord singleCsvRecord = SingleCSVRecord.fromRow(dataSet);
        fail(singleCsvRecord.toString() + " should have thrown an exception");
    }

    @Test
    public void testVectorOneHotLabel() {
        DataSet dataSet = new DataSet(Nd4j.create(2, 2), Nd4j.create(new double[][] {{0, 1}, {1, 0}}));

        //assert
        SingleCSVRecord singleCsvRecord = SingleCSVRecord.fromRow(dataSet.get(0));
        assertEquals(3, singleCsvRecord.getValues().size());

    }

    @Test
    public void testVectorRegression() {
        DataSet dataSet = new DataSet(Nd4j.create(2, 2), Nd4j.create(new double[][] {{1, 1}, {1, 1}}));

        //assert
        SingleCSVRecord singleCsvRecord = SingleCSVRecord.fromRow(dataSet.get(0));
        assertEquals(4, singleCsvRecord.getValues().size());

    }

}
