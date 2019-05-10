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

import org.datavec.spark.transform.model.SingleImageRecord;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;

/**
 * Created by kepricon on 17. 5. 24.
 */
public class SingleImageRecordTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testImageRecord() throws Exception {
        File f = testDir.newFolder();
        new ClassPathResource("datavec-spark-inference/testimages/").copyDirectory(f);
        File f0 = new File(f, "class0/0.jpg");
        File f1 = new File(f, "/class1/A.jpg");

        SingleImageRecord imgRecord = new SingleImageRecord(f0.toURI());

        // need jackson test?
    }
}
