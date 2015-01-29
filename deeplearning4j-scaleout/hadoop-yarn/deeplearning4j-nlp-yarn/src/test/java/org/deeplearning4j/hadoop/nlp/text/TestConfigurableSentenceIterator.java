package org.deeplearning4j.hadoop.nlp.text;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.io.InputStream;

/**
 * Created by agibsonccc on 1/29/15.
 */
public class TestConfigurableSentenceIterator extends ConfigurableSentenceIterator {


    public TestConfigurableSentenceIterator(Configuration conf) throws IOException {
        super(conf);
    }

    @Override
    public String nextSentence() {
        Path next = paths.next();
        try {
            InputStream open = fs.open(next);
            String read = new String(IOUtils.toByteArray(open));
            open.close();
            return read;
        }catch(Exception e) {

        }
        return null;
    }

    @Override
    public void reset() {

    }


}
