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

package org.datavec.api.split.parittion;

import org.junit.jupiter.api.Tag;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.shade.guava.io.Files;
import org.datavec.api.conf.Configuration;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.PartitionMetaData;
import org.datavec.api.split.partition.Partitioner;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.OutputStream;

import static org.junit.jupiter.api.Assertions.*;
@Tag(TagNames.JAVA_ONLY)
@Tag(TagNames.FILE_IO)
public class PartitionerTests extends BaseND4JTest {
    @Test
    public void testRecordsPerFilePartition() {
        Partitioner partitioner = new NumberOfRecordsPartitioner();
        File tmpDir = Files.createTempDir();
        FileSplit fileSplit = new FileSplit(tmpDir);
        assertTrue(fileSplit.needsBootstrapForWrite());
        fileSplit.bootStrapForWrite();
        partitioner.init(fileSplit);
        assertEquals(1,partitioner.numPartitions());
    }

    @Test
    public void testInputAddFile() throws Exception {
        Partitioner partitioner = new NumberOfRecordsPartitioner();
        File tmpDir = Files.createTempDir();
        FileSplit fileSplit = new FileSplit(tmpDir);
        assertTrue(fileSplit.needsBootstrapForWrite());
        fileSplit.bootStrapForWrite();
        Configuration configuration = new Configuration();
        configuration.set(NumberOfRecordsPartitioner.RECORDS_PER_FILE_CONFIG,String.valueOf(5));
        partitioner.init(configuration,fileSplit);
        partitioner.updatePartitionInfo(PartitionMetaData.builder().numRecordsUpdated(5).build());
        assertTrue(partitioner.needsNewPartition());
        OutputStream os = partitioner.openNewStream();
        os.close();
        assertNotNull(os);
        //run more than once to ensure output stream creation works properly
        partitioner.updatePartitionInfo(PartitionMetaData.builder().numRecordsUpdated(5).build());
        os = partitioner.openNewStream();
        os.close();
        assertNotNull(os);


    }

}
