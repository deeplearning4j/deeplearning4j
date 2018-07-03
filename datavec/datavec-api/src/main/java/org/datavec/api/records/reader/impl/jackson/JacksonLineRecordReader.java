package org.datavec.api.records.reader.impl.jackson;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.ObjectMapper;

/**
 * JacksonLineRecordReader will read a single file line-by-line when .next() is<br/>
 * called. It uses Jackson ObjectMapper and FieldSelection to read the fields in<br/>
 * each line.<br/>
 * <br/>
 * Each line should be a valid JSON entry without separator at the end. This is similar<br/>
 * to other readers and follows Hadoop convention. Hadoop and Spark use this format to<br/>
 * to make sure splits work properly in a cluster environment. For those new to Hadoop<br/>
 * file format convention, the reason is a large file can be split into chunks and<br/>
 * sent to different nodes in a cluster. If a record spanned multiple lines, split<br/>
 * might not get the complete record, which will result in runtime errors and calculation<br/>
 * errors. Where and how a job splits a file varies depending on the job configuration<br/>
 * and cluster size.<br/>
 * <br/>
 * A couple of important notes. The reader doesn't automatically create labels for each<br/>
 * record like JacksonRecordReader. JacksonRecordReader uses the folder name for the label<br/>
 * at runtime. It assumes a top level folder has multiple subfolders. The labels are the
 * subfolder names.<br/>
 * <br/>
 * In the case of JacksonLineRecordReader, you have to provide the labels in the configuration<br/>
 * for the training. Please look at the examples in dl4j-examples repository on how to provide
 * labels.<br/>
 * 
 * @author peter
 *
 */
public class JacksonLineRecordReader extends LineRecordReader {

    private FieldSelection selection;
    private ObjectMapper mapper;

	public JacksonLineRecordReader(FieldSelection selection, ObjectMapper mapper) {
		this.selection = selection;
		this.mapper = mapper;
	}

    @Override
    public List<Writable> next() {
        Text t = (Text) super.next().iterator().next();
        String val = t.toString();
        return parseLine(val);
    }
    
    protected List<Writable> parseLine(String line) {
	    return JacksonReaderUtils.parseRecord(line, selection, mapper);
    }
    
    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException();
    }
}
