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
package org.deeplearning4j.datasets.iterator;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.parallel.JointParallelDataSetIterator;
import org.deeplearning4j.datasets.iterator.tools.SimpleVariableGenerator;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.enums.InequalityHandling;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

@Slf4j
@DisplayName("Joint Parallel Data Set Iterator Test")
class JointParallelDataSetIteratorTest extends BaseDL4JTest {

    /**
     * Simple test, checking datasets alignment. They all should have the same data for the same cycle
     *
     * @throws Exception
     */
    @Test
    @DisplayName("Test Joint Iterator 1")
    void testJointIterator1() throws Exception {
        DataSetIterator iteratorA = new SimpleVariableGenerator(119, 100, 32, 100, 10);
        DataSetIterator iteratorB = new SimpleVariableGenerator(119, 100, 32, 100, 10);
        JointParallelDataSetIterator jpdsi = new JointParallelDataSetIterator.Builder(InequalityHandling.STOP_EVERYONE).addSourceIterator(iteratorA).addSourceIterator(iteratorB).build();
        int cnt = 0;
        int example = 0;
        while (jpdsi.hasNext()) {
            DataSet ds = jpdsi.next();
            assertNotNull(ds,"Failed on iteration " + cnt);
            // ds.detach();
            // ds.migrate();
            assertEquals( (double) example, ds.getFeatures().meanNumber().doubleValue(), 0.001,"Failed on iteration " + cnt);
            assertEquals( (double) example + 0.5, ds.getLabels().meanNumber().doubleValue(), 0.001,"Failed on iteration " + cnt);
            cnt++;
            if (cnt % 2 == 0)
                example++;
        }
        assertEquals(100, example);
        assertEquals(200, cnt);
    }

    /**
     * This test checks for pass_null scenario, so in total we should have 300 real datasets + 100 nulls
     * @throws Exception
     */
    @Test
    @DisplayName("Test Joint Iterator 2")
    void testJointIterator2() throws Exception {
        DataSetIterator iteratorA = new SimpleVariableGenerator(119, 200, 32, 100, 10);
        DataSetIterator iteratorB = new SimpleVariableGenerator(119, 100, 32, 100, 10);
        JointParallelDataSetIterator jpdsi = new JointParallelDataSetIterator.Builder(InequalityHandling.PASS_NULL).addSourceIterator(iteratorA).addSourceIterator(iteratorB).build();
        int cnt = 0;
        int example = 0;
        int nulls = 0;
        while (jpdsi.hasNext()) {
            DataSet ds = jpdsi.next();
            if (cnt < 200)
                assertNotNull(ds,"Failed on iteration " + cnt);
            if (ds == null)
                nulls++;
            if (cnt % 2 == 2) {
                assertEquals((double) example, ds.getFeatures().meanNumber().doubleValue(), 0.001,"Failed on iteration " + cnt);
                assertEquals((double) example + 0.5, ds.getLabels().meanNumber().doubleValue(), 0.001,"Failed on iteration " + cnt);
            }
            cnt++;
            if (cnt % 2 == 0)
                example++;
        }
        assertEquals(100, nulls);
        assertEquals(200, example);
        assertEquals(400, cnt);
    }

    /**
     * Testing relocate
     *
     * @throws Exception
     */
    @Test
    @DisplayName("Test Joint Iterator 3")
    void testJointIterator3() throws Exception {
        DataSetIterator iteratorA = new SimpleVariableGenerator(119, 200, 32, 100, 10);
        DataSetIterator iteratorB = new SimpleVariableGenerator(119, 100, 32, 100, 10);
        JointParallelDataSetIterator jpdsi = new JointParallelDataSetIterator.Builder(InequalityHandling.RELOCATE).addSourceIterator(iteratorA).addSourceIterator(iteratorB).build();
        int cnt = 0;
        int example = 0;
        while (jpdsi.hasNext()) {
            DataSet ds = jpdsi.next();
            assertNotNull(ds,"Failed on iteration " + cnt);
            assertEquals((double) example, ds.getFeatures().meanNumber().doubleValue(), 0.001,"Failed on iteration " + cnt);
            assertEquals( (double) example + 0.5, ds.getLabels().meanNumber().doubleValue(), 0.001,"Failed on iteration " + cnt);
            cnt++;
            if (cnt < 200) {
                if (cnt % 2 == 0)
                    example++;
            } else
                example++;
        }
        assertEquals(300, cnt);
        assertEquals(200, example);
    }

    /**
     * Testing relocate
     *
     * @throws Exception
     */
    @Test
    @DisplayName("Test Joint Iterator 4")
    void testJointIterator4() throws Exception {
        DataSetIterator iteratorA = new SimpleVariableGenerator(119, 200, 32, 100, 10);
        DataSetIterator iteratorB = new SimpleVariableGenerator(119, 100, 32, 100, 10);
        JointParallelDataSetIterator jpdsi = new JointParallelDataSetIterator.Builder(InequalityHandling.RESET).addSourceIterator(iteratorA).addSourceIterator(iteratorB).build();
        int cnt = 0;
        int cnt_sec = 0;
        int example_sec = 0;
        int example = 0;
        while (jpdsi.hasNext()) {
            DataSet ds = jpdsi.next();
            assertNotNull(ds,"Failed on iteration " + cnt);
            if (cnt % 2 == 0) {
                assertEquals( (double) example, ds.getFeatures().meanNumber().doubleValue(), 0.001,"Failed on iteration " + cnt);
                assertEquals((double) example + 0.5, ds.getLabels().meanNumber().doubleValue(), 0.001,"Failed on iteration " + cnt);
            } else {
                if (cnt <= 200) {
                    assertEquals((double) example, ds.getFeatures().meanNumber().doubleValue(), 0.001,"Failed on iteration " + cnt);
                    assertEquals( (double) example + 0.5, ds.getLabels().meanNumber().doubleValue(), 0.001,"Failed on iteration " + cnt);
                } else {
                    assertEquals((double) example_sec, ds.getFeatures().meanNumber().doubleValue(), 0.001,"Failed on iteration " + cnt + ", second iteration " + cnt_sec);
                    assertEquals((double) example_sec + 0.5, ds.getLabels().meanNumber().doubleValue(), 0.001,"Failed on iteration " + cnt + ", second iteration " + cnt_sec);
                }
            }
            cnt++;
            if (cnt % 2 == 0)
                example++;
            if (cnt > 201 && cnt % 2 == 1) {
                cnt_sec++;
                example_sec++;
            }
        }
        assertEquals(400, cnt);
        assertEquals(200, example);
    }
}
