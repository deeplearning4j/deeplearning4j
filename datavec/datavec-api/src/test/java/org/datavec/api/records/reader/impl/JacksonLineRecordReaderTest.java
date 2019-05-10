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

package org.datavec.api.records.reader.impl;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.jackson.FieldSelection;
import org.datavec.api.records.reader.impl.jackson.JacksonLineRecordReader;
import org.datavec.api.records.reader.impl.jackson.JacksonLineSequenceRecordReader;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.File;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class JacksonLineRecordReaderTest {

	@Rule
	public TemporaryFolder testDir = new TemporaryFolder();

	public JacksonLineRecordReaderTest() {
	}

    private static FieldSelection getFieldSelection() {
        return new FieldSelection.Builder().addField("value1").
        		addField("value2").
        		addField("value3").
        		addField("value4").
        		addField("value5").
        		addField("value6").
        		addField("value7").
        		addField("value8").
        		addField("value9").
        		addField("value10").build();
    }
    
    @Test
    public void testReadJSON() throws Exception {
       
        RecordReader rr = new JacksonLineRecordReader(getFieldSelection(), new ObjectMapper(new JsonFactory()));
        rr.initialize(new FileSplit(new ClassPathResource("datavec-api/json/json_test_3.txt").getFile()));
        
        testJacksonRecordReader(rr);
	}
    
    private static void testJacksonRecordReader(RecordReader rr) {
    	while (rr.hasNext()) {
        	List<Writable> json0 = rr.next();
        	//System.out.println(json0);
        	assert(json0.size() > 0);
    	}
    }


    @Test
	public void testJacksonLineSequenceRecordReader() throws Exception {
		File dir = testDir.newFolder();
		new ClassPathResource("datavec-api/JacksonLineSequenceRecordReaderTest/").copyDirectory(dir);

		FieldSelection f = new FieldSelection.Builder().addField("a").addField(new Text("MISSING_B"), "b")
				.addField(new Text("MISSING_CX"), "c", "x").build();

		JacksonLineSequenceRecordReader rr = new JacksonLineSequenceRecordReader(f, new ObjectMapper(new JsonFactory()));
		File[] files = dir.listFiles();
		Arrays.sort(files);
		URI[] u = new URI[files.length];
		for( int i=0; i<files.length; i++ ){
			u[i] = files[i].toURI();
		}
		rr.initialize(new CollectionInputSplit(u));

		List<List<Writable>> expSeq0 = new ArrayList<>();
		expSeq0.add(Arrays.asList((Writable) new Text("aValue0"), new Text("bValue0"), new Text("cxValue0")));
		expSeq0.add(Arrays.asList((Writable) new Text("aValue1"), new Text("MISSING_B"), new Text("cxValue1")));
		expSeq0.add(Arrays.asList((Writable) new Text("aValue2"), new Text("bValue2"), new Text("MISSING_CX")));

		List<List<Writable>> expSeq1 = new ArrayList<>();
		expSeq1.add(Arrays.asList((Writable) new Text("aValue3"), new Text("bValue3"), new Text("cxValue3")));


		int count = 0;
		while(rr.hasNext()){
			List<List<Writable>> next = rr.sequenceRecord();
			if(count++ == 0){
				assertEquals(expSeq0, next);
			} else {
				assertEquals(expSeq1, next);
			}
		}

		assertEquals(2, count);
	}
}
