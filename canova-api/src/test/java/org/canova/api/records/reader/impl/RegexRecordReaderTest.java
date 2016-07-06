package org.canova.api.records.reader.impl;

import org.canova.api.io.data.Text;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.records.reader.impl.regex.RegexLineRecordReader;
import org.canova.api.records.reader.impl.regex.RegexSequenceRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.split.NumberedFileInputSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Created by Alex on 12/04/2016.
 */
public class RegexRecordReaderTest {

    @Test
    public void testRegexLineRecordReader() throws Exception {
        String regex = "(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d{3}) (\\d+) ([A-Z]+) (.*)";

        RecordReader rr = new RegexLineRecordReader(regex, 1);
        rr.initialize(new FileSplit(new ClassPathResource("/logtestdata/logtestfile0.txt").getFile()));

        List<Writable> exp0 = Arrays.asList((Writable) new Text("2016-01-01 23:59:59.001"), new Text("1"), new Text("DEBUG"), new Text("First entry message!"));
        List<Writable> exp1 = Arrays.asList((Writable) new Text("2016-01-01 23:59:59.002"), new Text("2"), new Text("INFO"), new Text("Second entry message!"));
        List<Writable> exp2 = Arrays.asList((Writable) new Text("2016-01-01 23:59:59.003"), new Text("3"), new Text("WARN"), new Text("Third entry message!"));
        assertEquals(exp0, rr.next());
        assertEquals(exp1, rr.next());
        assertEquals(exp2, rr.next());
        assertFalse(rr.hasNext());

        //Test reset:
        rr.reset();
        assertEquals(exp0, rr.next());
        assertEquals(exp1, rr.next());
        assertEquals(exp2, rr.next());
        assertFalse(rr.hasNext());
    }

    @Test
    public void testRegexSequenceRecordReader() throws Exception {
        String regex = "(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d{3}) (\\d+) ([A-Z]+) (.*)";

        String path = new ClassPathResource("/logtestdata/logtestfile0.txt").getFile().getAbsolutePath();
        path = path.replace("0","%d");

        InputSplit is = new NumberedFileInputSplit(path,0,1);

        SequenceRecordReader rr = new RegexSequenceRecordReader(regex,1);
        rr.initialize(is);

        Collection<Collection<Writable>> exp0 = new ArrayList<>();
        exp0.add(Arrays.asList((Writable) new Text("2016-01-01 23:59:59.001"), new Text("1"), new Text("DEBUG"), new Text("First entry message!")));
        exp0.add(Arrays.asList((Writable) new Text("2016-01-01 23:59:59.002"), new Text("2"), new Text("INFO"), new Text("Second entry message!")));
        exp0.add(Arrays.asList((Writable) new Text("2016-01-01 23:59:59.003"), new Text("3"), new Text("WARN"), new Text("Third entry message!")));


        Collection<Collection<Writable>> exp1 = new ArrayList<>();
        exp1.add(Arrays.asList((Writable) new Text("2016-01-01 23:59:59.011"), new Text("11"), new Text("DEBUG"), new Text("First entry message!")));
        exp1.add(Arrays.asList((Writable) new Text("2016-01-01 23:59:59.012"), new Text("12"), new Text("INFO"), new Text("Second entry message!")));
        exp1.add(Arrays.asList((Writable) new Text("2016-01-01 23:59:59.013"), new Text("13"), new Text("WARN"), new Text("Third entry message!")));

        assertEquals(exp0,rr.sequenceRecord());
        assertEquals(exp1,rr.sequenceRecord());
        assertFalse(rr.hasNext());

        //Test resetting:
        rr.reset();
        assertEquals(exp0,rr.sequenceRecord());
        assertEquals(exp1,rr.sequenceRecord());
        assertFalse(rr.hasNext());
    }

}
