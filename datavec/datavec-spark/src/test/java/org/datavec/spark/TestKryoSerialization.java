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

package org.datavec.spark;

import org.apache.spark.serializer.KryoSerializerInstance;
import org.apache.spark.serializer.SerializerInstance;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.tests.tags.TagNames;

import java.io.File;
import java.nio.ByteBuffer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
@Tag(TagNames.FILE_IO)
@Tag(TagNames.JAVA_ONLY)
@Tag(TagNames.SPARK)
@Tag(TagNames.DIST_SYSTEMS)
public class TestKryoSerialization extends BaseSparkTest {

    @Override
    public boolean useKryo(){
        return true;
    }

    @Test
    public void testCsvRecordReader() throws Exception {
        SerializerInstance si = sc.env().serializer().newInstance();
        assertTrue(si instanceof KryoSerializerInstance);

        RecordReader r1 = new CSVRecordReader(1,'\t');
        RecordReader r2 = serDe(r1, si);

        File f = new ClassPathResource("iris_tab_delim.txt").getFile();
        r1.initialize(new FileSplit(f));
        r2.initialize(new FileSplit(f));

        while(r1.hasNext()){
            assertEquals(r1.next(), r2.next());
        }
        assertFalse(r2.hasNext());
    }


    private <T> T serDe(T in, SerializerInstance si){
        ByteBuffer bb = si.serialize(in, null);
        return (T)si.deserialize(bb, null);
    }
}
