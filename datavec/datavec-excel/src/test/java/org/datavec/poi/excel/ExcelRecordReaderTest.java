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
package org.datavec.poi.excel;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.io.ClassPathResource;
import java.util.List;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

@DisplayName("Excel Record Reader Test")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.JAVA_ONLY)
class ExcelRecordReaderTest {

    @Test
    @DisplayName("Test Simple")
    void testSimple() throws Exception {
        RecordReader excel = new ExcelRecordReader();
        excel.initialize(new FileSplit(new ClassPathResource("datavec-excel/testsheet.xlsx").getFile()));
        assertTrue(excel.hasNext());
        List<Writable> next = excel.next();
        assertEquals(3, next.size());
        RecordReader headerReader = new ExcelRecordReader(1);
        headerReader.initialize(new FileSplit(new ClassPathResource("datavec-excel/testsheetheader.xlsx").getFile()));
        assertTrue(excel.hasNext());
        List<Writable> next2 = excel.next();
        assertEquals(3, next2.size());
    }
}
