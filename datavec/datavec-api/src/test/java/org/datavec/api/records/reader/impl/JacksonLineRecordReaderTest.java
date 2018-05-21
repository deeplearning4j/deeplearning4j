package org.datavec.api.records.reader.impl;

import java.util.List;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.jackson.FieldSelection;
import org.datavec.api.records.reader.impl.jackson.JacksonLineRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.databind.ObjectMapper;

public class JacksonLineRecordReaderTest {

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
        rr.initialize(new FileSplit(new ClassPathResource("json/json_test_3.txt").getFile()));
        
        testJacksonRecordReader(rr);
	}
    
    private static void testJacksonRecordReader(RecordReader rr) {
    	while (rr.hasNext()) {
        	List<Writable> json0 = rr.next();
        	//System.out.println(json0);
        	assert(json0.size() > 0);
    	}
    }
}
