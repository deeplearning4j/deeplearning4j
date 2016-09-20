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

package org.datavec.api.records.reader.impl;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.datavec.api.berkeley.Pair;
import org.datavec.api.berkeley.Triple;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataLine;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.split.StringSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.io.*;
import java.lang.reflect.Array;
import java.net.URI;
import java.util.*;

/**
 * Reads files line by line
 *
 * @author Adam Gibson
 */
public class LineRecordReader extends BaseRecordReader {


    private Iterator<String> iter;
    protected URI[] locations;
    protected int currIndex = 0;
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
    public List<Writable> next() {
        List<Writable> ret = new ArrayList<>();

        if(iter.hasNext()) {
            String record = iter.next();
            invokeListeners(record);
            ret.add(new Text(record));
            currIndex++;
            return ret;
        } else {
            if ( !(inputSplit instanceof StringSplit) && currIndex < locations.length-1 ) {
                currIndex++;
                try {
                    close();
                    iter = IOUtils.lineIterator(new InputStreamReader(locations[currIndex].toURL().openStream()));
                    onLocationOpen(locations[currIndex]);
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
                    onLocationOpen(locations[currIndex]);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                return iter.hasNext();
            }

            return false;
        }
    }

    protected void onLocationOpen(URI location) {

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
            currIndex = 0;
        }catch(Exception e){
            throw new RuntimeException("Error during LineRecordReader reset",e);
        }
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        invokeListeners(uri);
        //Here: we are reading a single line from the DataInputStream
        BufferedReader br = new BufferedReader(new InputStreamReader(dataInputStream));
        String line = br.readLine();
        return Collections.singletonList((Writable)new Text(line));
    }


    //@Override //TODO
    public List<Record> loadFromMeta(List<RecordMetaData> recordMetaDatas) throws IOException {
        //First: create a sorted list of the RecordMetaData
        List<Triple<Integer,RecordMetaDataLine,List<Writable>>> list = new ArrayList<>();
        Iterator<RecordMetaData> iter = recordMetaDatas.iterator();
        int count = 0;
        while(iter.hasNext()){
            RecordMetaData rmd = iter.next();
            if(!(rmd instanceof RecordMetaDataLine)){
                throw new IllegalArgumentException("Invalid metadata; expected RecordMetaDataLine instance; got: " + rmd);
            }
            list.add(new Triple<>(count++, (RecordMetaDataLine)rmd, (List<Writable>)null));
        }

//        Collections.sort(list, new Comparator<Pair<Integer,RecordMetaData>>(){
//            @Override
//            public int compare(Pair<Integer, RecordMetaData> o1, Pair<Integer, RecordMetaData> o2) {
//                return Integer.compare(o1.getFirst(), o2.getFirst());
//            }
//        });
        //Sort by line number:
        Collections.sort(list, new Comparator<Triple<Integer,RecordMetaDataLine, List<Writable>>>(){
            @Override
            public int compare(Triple<Integer, RecordMetaDataLine, List<Writable>> o1, Triple<Integer, RecordMetaDataLine, List<Writable>> o2) {
                return Integer.compare(o1.getSecond().getLineNumber(), o2.getSecond().getLineNumber());
            }
        });

        Iterator<String> iterator = null;
        if(inputSplit instanceof StringSplit) {
            StringSplit stringSplit = (StringSplit) inputSplit;
            iterator = Collections.singletonList(stringSplit.getData()).listIterator();
        } else if (inputSplit instanceof InputStreamInputSplit){
            InputStream is = ((InputStreamInputSplit) inputSplit).getIs();
            if(is != null){
                iterator =  IOUtils.lineIterator(new InputStreamReader(is));
            }
        } else {
            this.locations = inputSplit.locations();
            if (locations != null && locations.length > 0) {
                iterator =  IOUtils.lineIterator(new InputStreamReader(locations[0].toURL().openStream()));
            }
        }
        if(iterator == null) throw new UnsupportedOperationException(); //TODO

        Iterator<Triple<Integer,RecordMetaDataLine,List<Writable>>> metaIter = list.iterator();
        int currentLineIdx = 0;
        String line = iterator.next();
        while(metaIter.hasNext()){
            Triple<Integer,RecordMetaDataLine,List<Writable>> t = metaIter.next();
            int nextLineIdx = t.getSecond().getLineNumber();
            while(currentLineIdx < nextLineIdx && iterator.hasNext()){
                line = iterator.next();
                currentLineIdx++;
            }
            t.setThird(Collections.<Writable>singletonList(new Text(line)));
        }

        //Now, sort by the original order:
        Collections.sort(list, new Comparator<Triple<Integer,RecordMetaDataLine, List<Writable>>>(){
            @Override
            public int compare(Triple<Integer, RecordMetaDataLine, List<Writable>> o1, Triple<Integer, RecordMetaDataLine, List<Writable>> o2) {
                return Integer.compare(o1.getFirst(), o2.getFirst());
            }
        });

        //And return...
        List<Record> out = new ArrayList<>();
        for(Triple<Integer,RecordMetaDataLine,List<Writable>> t : list){
            out.add(new Record(t.getThird(), t.getSecond()));
        }
        return out;
    }
}
