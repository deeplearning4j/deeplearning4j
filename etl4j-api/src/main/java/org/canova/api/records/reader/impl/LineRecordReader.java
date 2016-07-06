/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.api.records.reader.impl;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.canova.api.conf.Configuration;
import org.canova.api.io.data.Text;
import org.canova.api.records.reader.BaseRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.split.InputStreamInputSplit;
import org.canova.api.split.StringSplit;
import org.canova.api.writable.Writable;

import java.io.*;
import java.net.URI;
import java.util.*;

/**
 * Reads files line by line
 *
 * @author Adam Gibson
 */
public class LineRecordReader extends BaseRecordReader {


    private Iterator<String> iter;
    private URI[] locations;
    private int currIndex = 0;
    protected Configuration conf;
    protected InputSplit inputSplit;

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        if(split instanceof StringSplit) {
            StringSplit stringSplit = (StringSplit) split;
            iter = Arrays.asList(stringSplit.getData()).listIterator();
        } else if (split instanceof InputStreamInputSplit){
            InputStream is = ((InputStreamInputSplit) split).getIs();
            if(is != null){
                iter =  IOUtils.lineIterator(new InputStreamReader(is));
            }
        } else {
            this.locations = split.locations();
            if (locations != null && locations.length > 0) {
                iter =  IOUtils.lineIterator(new InputStreamReader(locations[0].toURL().openStream()));
            }
        }
        this.inputSplit = split;
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        initialize(split);
    }

    @Override
    public Collection<Writable> next() {
        List<Writable> ret = new ArrayList<>();

        if(iter.hasNext()) {
            String record = iter.next();
            invokeListeners(record);
            ret.add(new Text(record));
            return ret;
        } else {
            if ( !(inputSplit instanceof StringSplit) && currIndex < locations.length-1 ) {
                currIndex++;
                try {
                    close();
                    iter = IOUtils.lineIterator(new InputStreamReader(locations[currIndex].toURL().openStream()));
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if(iter.hasNext()) {
                    String record = iter.next();
                    invokeListeners(record);
                    ret.add(new Text(record));
                    return ret;
                }
            }

            throw new NoSuchElementException("No more elements found!");
        }
    }

    @Override
    public boolean hasNext() {
        if ( iter != null && iter.hasNext() ) {
            return true;
        } else {
            if (locations != null && !(inputSplit instanceof StringSplit) && currIndex < locations.length-1 ) {
                currIndex++;
                try {
                    close();
                    iter = IOUtils.lineIterator(new InputStreamReader(locations[currIndex].toURL().openStream()));
                } catch (IOException e) {
                    e.printStackTrace();
                }

                return iter.hasNext();
            }

            return false;
        }
    }

    @Override
    public void close() throws IOException {
        if(iter != null) {
            if(iter instanceof LineIterator) {
                LineIterator iter2 = (LineIterator) iter;
                iter2.close();
            }
        }
    }

    @Override
    public void setConf(Configuration conf) {
       this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    @Override
    public List<String> getLabels(){
        return null; }

    @Override
    public void reset() {
        if(inputSplit == null) throw new UnsupportedOperationException("Cannot reset without first initializing");
        try{
            initialize(inputSplit);
        }catch(Exception e){
            throw new RuntimeException("Error during LineRecordReader reset",e);
        }
    }

    @Override
    public Collection<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        invokeListeners(uri);
        //Here: we are reading a single line from the DataInputStream
        BufferedReader br = new BufferedReader(new InputStreamReader(dataInputStream));
        String line = br.readLine();
        return Collections.singletonList((Writable)new Text(line));
    }
}
