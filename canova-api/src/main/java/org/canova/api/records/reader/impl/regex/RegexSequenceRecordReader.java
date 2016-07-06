package org.canova.api.records.reader.impl.regex;

import org.apache.commons.io.FileUtils;
import org.canova.api.conf.Configuration;
import org.canova.api.io.data.Text;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.records.reader.impl.FileRecordReader;
import org.canova.api.records.reader.impl.LineRecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.misc.IOUtils;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * RegexSequenceRecordReader: Read an entire file (as a sequence), one line at a time and
 * split each line into fields using a regex.
 * Specifically, we are using {@link Pattern} and {@link Matcher} to do the splitting into groups
 *
 * Example: Data in format "2016-01-01 23:59:59.001 1 DEBUG First entry message!"<br>
 * using regex String "(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d{3}) (\\d+) ([A-Z]+) (.*)"<br>
 * would be split into 4 Text writables: ["2016-01-01 23:59:59.001", "1", "DEBUG", "First entry message!"]<br>
 *
 * Note: RegexSequenceRecordReader supports multiple error handling modes, via {@link LineErrorHandling}. Invalid
 * lines that don't match the provided regex can result in an exception (FailOnInvalid), can be skipped silently (SkipInvalid),
 * or skip invalid but log a warning (SkipInvalidWithWarning)
 *
 * @author Alex Black
 */
public class RegexSequenceRecordReader extends FileRecordReader implements SequenceRecordReader {
    public static final String SKIP_NUM_LINES = NAME_SPACE + ".skipnumlines";
    public static final Charset DEFAULT_CHARSET = Charset.forName("UTF-8");
    public static final LineErrorHandling DEFAULT_ERROR_HANDLING = LineErrorHandling.FailOnInvalid;

    /**Error handling mode: How should invalid lines (i.e., those that don't match the provided regex) be handled?<br>
     * FailOnInvalid: Throw an IllegalStateException when an invalid line is found<br>
     * SkipInvalid: Skip invalid lines (quietly, with no warning)<br>
     * SkipInvalidWithWarning: Skip invalid lines, but log a warning<br>
     */
    public enum LineErrorHandling {FailOnInvalid, SkipInvalid, SkipInvalidWithWarning};

    public static final Logger LOG = LoggerFactory.getLogger(RegexSequenceRecordReader.class);

    private String regex;
    private int skipNumLines;
    private Pattern pattern;
    private Charset charset;
    private LineErrorHandling errorHandling;

    public RegexSequenceRecordReader(String regex, int skipNumLines){
        this(regex, skipNumLines, DEFAULT_CHARSET, DEFAULT_ERROR_HANDLING);
    }

    public RegexSequenceRecordReader(String regex, int skipNumLines, Charset encoding, LineErrorHandling errorHandling){
        this.regex = regex;
        this.skipNumLines = skipNumLines;
        this.pattern = Pattern.compile(regex);
        this.charset = encoding;
        this.errorHandling = errorHandling;
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        this.skipNumLines = conf.getInt(SKIP_NUM_LINES,this.skipNumLines);
    }

    public Collection<Collection<Writable>> sequenceRecord() {
        File next = iter.next();

        String fileContents;
        try {
            fileContents = FileUtils.readFileToString(next, charset.name());
        } catch(IOException e){
            throw new RuntimeException(e);
        }
        return loadSequence(fileContents, next.toURI());
    }

    @Override
    public
    Collection<Collection<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        String fileContents = org.apache.commons.io.IOUtils.toString(dataInputStream,charset.name());
        return loadSequence(fileContents, uri);
    }

    private Collection<Collection<Writable>> loadSequence(String fileContents, URI uri){
        String[] lines = fileContents.split("(\r\n)|\n");  //TODO this won't work if regex allows for a newline

        int numLinesSkipped = 0;
        Collection<Collection<Writable>> out = new ArrayList<>();
        int lineCount = 0;
        for(String line : lines){
            lineCount++;
            if(numLinesSkipped < skipNumLines){
                numLinesSkipped++;
                continue;
            }
            //Split line using regex matcher
            Matcher m = pattern.matcher(line);
            List<Writable> timeStep;
            if(m.matches()){
                int count = m.groupCount();
                timeStep = new ArrayList<>(count);
                for( int i=1; i<=count; i++){    //Note: Matcher.group(0) is the entire sequence; we only care about groups 1 onward
                    timeStep.add(new Text(m.group(i)));
                }
            } else {
                switch(errorHandling){
                    case FailOnInvalid:
                        throw new IllegalStateException("Invalid line: line does not match regex (line #" + lineCount + ", uri=\"" +
                                uri + "\"), " + "\", regex=" + regex + "\"; line=\"" + line + "\"");
                    case SkipInvalid:
                        continue;
                    case SkipInvalidWithWarning:
                        String warnMsg = "Skipping invalid line: line does not match regex (line #" + lineCount + ", uri=\"" +
                                uri + "\"), " + "\"; line=\"" + line + "\"";
                        LOG.warn(warnMsg);
                        continue;
                    default:
                        throw new RuntimeException("Unknown error handling mode: " + errorHandling);
                }
            }
            out.add(timeStep);
        }

        return out;
    }

    @Override
    public void reset(){
        super.reset();
    }

}
