package org.canova.api.records.reader.impl.regex;

import org.canova.api.conf.Configuration;
import org.canova.api.io.data.Text;
import org.canova.api.records.reader.impl.LineRecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * RegexLineRecordReader: Read a file, one line at a time, and split it into fields using a regex.
 * Specifically, we are using {@link java.util.regex.Pattern} and {@link java.util.regex.Matcher}.<br>
 * To load an entire file using a
 *
 * Example: Data in format "2016-01-01 23:59:59.001 1 DEBUG First entry message!"<br>
 * using regex String "(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d{3}) (\\d+) ([A-Z]+) (.*)"<br>
 * would be split into 4 Text writables: ["2016-01-01 23:59:59.001", "1", "DEBUG", "First entry message!"]
 *
 * @author Alex Black
 */
public class RegexLineRecordReader extends LineRecordReader {
    public final static String SKIP_NUM_LINES = NAME_SPACE + ".skipnumlines";

    private String regex;
    private int skipNumLines;
    private Pattern pattern;
    private int numLinesSkipped;
    private int currLine = 0;

    public RegexLineRecordReader(String regex, int skipNumLines){
        this.regex = regex;
        this.skipNumLines = skipNumLines;
        this.pattern = Pattern.compile(regex);
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        this.skipNumLines = conf.getInt(SKIP_NUM_LINES,this.skipNumLines);
    }

    @Override
    public Collection<Writable> next() {

        if(numLinesSkipped < skipNumLines) {
            for(int i = numLinesSkipped; i < skipNumLines; i++, numLinesSkipped++) {
                if(!hasNext()) {
                    return new ArrayList<>();
                }
                super.next();
            }
        }
        Text t =  (Text) super.next().iterator().next();
        String val = t.toString();
        return getRecord(val);
    }

    @Override
    public Collection<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        Writable w = ((List<Writable>)super.record(uri,dataInputStream)).get(0);
        return getRecord(w.toString());
    }

    private Collection<Writable> getRecord(String line){
        Matcher m = pattern.matcher(line);

        List<Writable> ret;
        if(m.matches()){
            int count = m.groupCount();
            ret = new ArrayList<>(count);
            for( int i=1; i<=count; i++){    //Note: Matcher.group(0) is the entire sequence; we only care about groups 1 onward
                ret.add(new Text(m.group(i)));
            }
        } else {
            throw new IllegalStateException("Invalid line: line does not match regex (line #" + currLine  +", regex=\""
                    + regex + "\"; line=\"" + line + "\"");
        }

        return ret;
    }

    @Override
    public void reset(){
        super.reset();
        numLinesSkipped = 0;
    }

}
