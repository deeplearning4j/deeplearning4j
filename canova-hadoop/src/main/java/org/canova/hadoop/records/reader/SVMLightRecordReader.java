package org.canova.hadoop.records.reader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.StringTokenizer;

import org.canova.api.conf.Configuration;
import org.canova.api.io.data.DoubleWritable;
import org.canova.api.io.data.Text;
import org.canova.api.records.reader.impl.LineRecordReader;
import org.canova.api.split.InputSplit;
//import org.canova.api.records.reader.impl.SVMLightRecordReader;
import org.canova.api.writable.Writable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SVMLightRecordReader extends LineRecordReader {
	
    private static Logger log = LoggerFactory.getLogger(SVMLightRecordReader.class);

    public SVMLightRecordReader() {
    }
    
    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        initialize(split);
    }

    @Override
    public boolean hasNext() {
    //    return iter != null && iter.hasNext();
    	return false;
    }    

   
    /**
     * next() method for getting another K/V pair off disk from the SVMLight text file
     * 
     */
    @Override
    public Collection<Writable> next() {
    	
        Text t =  (Text) super.next().iterator().next();
        
        
        String val = new String(t.getBytes());
        Collection<Writable> ret = new ArrayList<>();
        StringTokenizer tok;
        int	index,max;
        String	col;
        double	value;

        // actual data
        try {
            // determine max index
            max = 0;
            tok = new StringTokenizer(val, " \t");
            tok.nextToken();  // skip class
            while (tok.hasMoreTokens()) {
                col = tok.nextToken();
                // finished?
                if (col.startsWith("#"))
                    break;
                // qid is not supported
                if (col.startsWith("qid:"))
                    continue;
                // actual value
                index = Integer.parseInt(col.substring(0, col.indexOf(":")));
                if (index > max)
                    max = index;
            }

            // read values into array
            tok    = new StringTokenizer(val, " \t");

            // 1. class
            double classVal = Double.parseDouble(tok.nextToken());

            // 2. attributes
            while (tok.hasMoreTokens()) {
                col  = tok.nextToken();
                // finished?
                if (col.startsWith("#"))
                    break;
                // qid is not supported
                if (col.startsWith("qid:"))
                    continue;
                // actual value
                index = Integer.parseInt(col.substring(0, col.indexOf(":")));
                value = Double.parseDouble(col.substring(col.indexOf(":") + 1));
                ret.add(new DoubleWritable(value));
            }

            ret.add(new DoubleWritable(classVal));
        }
        catch (Exception e) {
            log.error("Error parsing line '" + val + "': ",e);
        }

        return ret;
    }


}
