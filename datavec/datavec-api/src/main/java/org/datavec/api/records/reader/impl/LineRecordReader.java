/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.records.reader.impl;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
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
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Triple;

import java.io.*;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Reads files line by line
 *
 * @author Adam Gibson
 */
@Slf4j
public class LineRecordReader extends BaseRecordReader {


    private Iterator<String> iter;
    protected URI[] locations;
    protected int splitIndex = 0;
    protected int lineIndex = 0; //Line index within the current split
    protected Configuration conf;
    protected boolean initialized;
    @Getter @Setter
    protected String charset = StandardCharsets.UTF_8.name(); //Using String as StandardCharsets.UTF_8 is not serializable

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        super.initialize(split);
        if(!(inputSplit instanceof StringSplit || inputSplit instanceof InputStreamInputSplit)){
            final ArrayList<URI> uris = new ArrayList<>();
            final Iterator<URI> uriIterator = inputSplit.locationsIterator();
            while(uriIterator.hasNext()) uris.add(uriIterator.next());

            this.locations = uris.toArray(new URI[0]);
        }
        this.iter = getIterator(0);
        this.initialized = true;
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        this.conf = conf;
        initialize(split);
    }

    @Override
    public List<Writable> next() {
        Preconditions.checkState(initialized, "Record reader has not been initialized");
        List<Writable> ret = new ArrayList<>();

        if (iter.hasNext()) {
            String record = iter.next();
            invokeListeners(record);
            ret.add(new Text(record));
            lineIndex++;
            return ret;
        } else {
            if (!(inputSplit instanceof StringSplit) && splitIndex < locations.length - 1) {
                splitIndex++;
                lineIndex = 0; //New split opened -> reset line index
                try {
                    close();
                    iter = getIterator(splitIndex);
                    onLocationOpen(locations[splitIndex]);
                } catch (IOException e) {
                    log.error("",e);
                }

                if (iter.hasNext()) {
                    String record = iter.next();
                    invokeListeners(record);
                    ret.add(new Text(record));
                    lineIndex++;
                    return ret;
                }
            }

            throw new NoSuchElementException("No more elements found!");
        }
    }

    @Override
    public boolean hasNext() {
        Preconditions.checkState(initialized, "Record reader has not been initialized");

        if (iter != null && iter.hasNext()) {
            return true;
        } else {
            if (locations != null && !(inputSplit instanceof StringSplit) && splitIndex < locations.length - 1) {
                splitIndex++;
                lineIndex = 0; //New split -> reset line count
                try {
                    close();
                    iter = getIterator(splitIndex);
                    onLocationOpen(locations[splitIndex]);
                } catch (IOException e) {
                    log.error("",e);
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
        if (iter != null) {
            if (iter instanceof LineIterator) {
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
    public List<String> getLabels() {
        return null;
    }

    @Override
    public void reset() {
        if (inputSplit == null)
            throw new UnsupportedOperationException("Cannot reset without first initializing");
        try {
            inputSplit.reset();
            close();
            initialize(inputSplit);
            splitIndex = 0;
        } catch (Exception e) {
            throw new RuntimeException("Error during LineRecordReader reset", e);
        }
        lineIndex = 0;
    }

    @Override
    public boolean resetSupported() {
        if(inputSplit != null){
            return inputSplit.resetSupported();
        }
        return true;
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        invokeListeners(uri);
        //Here: we are reading a single line from the DataInputStream
        BufferedReader br = new BufferedReader(new InputStreamReader(dataInputStream));
        String line = br.readLine();
        return Collections.singletonList((Writable) new Text(line));
    }

    protected Iterator<String> getIterator(int location) {
        Iterator<String> iterator = null;
        if (inputSplit instanceof StringSplit) {
            StringSplit stringSplit = (StringSplit) inputSplit;
            iterator = Collections.singletonList(stringSplit.getData()).listIterator();
        } else if (inputSplit instanceof InputStreamInputSplit) {
            InputStream is = ((InputStreamInputSplit) inputSplit).getIs();
            if (is != null) {
                try {
                    iterator = IOUtils.lineIterator(new InputStreamReader(is, charset));
                } catch (UnsupportedEncodingException e){
                    throw new RuntimeException("Unsupported encoding: " + charset, e);
                }
            }
        } else {
            if (locations.length > 0) {
                InputStream inputStream = streamCreatorFn.apply(locations[location]);
                try {
                    iterator = IOUtils.lineIterator(new InputStreamReader(inputStream, charset));
                } catch (UnsupportedEncodingException e){
                    throw new RuntimeException("Unsupported encoding: " + charset, e);
                }
            }
        }
        if (iterator == null)
            throw new UnsupportedOperationException("Unknown input split: " + inputSplit);
        return iterator;
    }

    protected void closeIfRequired(Iterator<String> iterator) {
        if (iterator instanceof LineIterator) {
            LineIterator iter = (LineIterator) iterator;
            iter.close();
        }
    }

    @Override
    public Record nextRecord() {
        List<Writable> next = next();
        URI uri = (locations == null || locations.length < 1 ? null : locations[splitIndex]);
        RecordMetaData meta = new RecordMetaDataLine(this.lineIndex - 1, uri, LineRecordReader.class); //-1 as line number has been incremented already...
        return new org.datavec.api.records.impl.Record(next, meta);
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return null;
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        //First: create a sorted list of the RecordMetaData
        List<Triple<Integer, RecordMetaDataLine, List<Writable>>> list = new ArrayList<>();
        Set<URI> uris = new HashSet<>();
        Iterator<RecordMetaData> iter = recordMetaDatas.iterator();
        int count = 0;
        while (iter.hasNext()) {
            RecordMetaData rmd = iter.next();
            if (!(rmd instanceof RecordMetaDataLine)) {
                throw new IllegalArgumentException(
                                "Invalid metadata; expected RecordMetaDataLine instance; got: " + rmd);
            }
            list.add(new Triple<>(count++, (RecordMetaDataLine) rmd, (List<Writable>) null));
            if (rmd.getURI() != null)
                uris.add(rmd.getURI());
        }
        List<URI> sortedURIs = null;
        if (uris.size() > 0) {
            sortedURIs = new ArrayList<>(uris);
            Collections.sort(sortedURIs);
        }

        //Sort by URI first (if possible - don't always have URIs though, for String split etc), then sort by line number:
        Collections.sort(list, new Comparator<Triple<Integer, RecordMetaDataLine, List<Writable>>>() {
            @Override
            public int compare(Triple<Integer, RecordMetaDataLine, List<Writable>> o1,
                            Triple<Integer, RecordMetaDataLine, List<Writable>> o2) {
                if (o1.getSecond().getURI() != null) {
                    if (!o1.getSecond().getURI().equals(o2.getSecond().getURI())) {
                        return o1.getSecond().getURI().compareTo(o2.getSecond().getURI());
                    }
                }
                return Integer.compare(o1.getSecond().getLineNumber(), o2.getSecond().getLineNumber());
            }
        });

        if (uris.size() > 0 && sortedURIs != null) {
            //URIs case - possibly with multiple URIs
            Iterator<Triple<Integer, RecordMetaDataLine, List<Writable>>> metaIter = list.iterator(); //Currently sorted by URI, then line number

            URI currentURI = sortedURIs.get(0);
            Iterator<String> currentUriIter = IOUtils.lineIterator(streamCreatorFn.apply(currentURI), charset);

            int currentURIIdx = 0; //Index of URI
            int currentLineIdx = 0; //Index of the line for the current URI
            String line = currentUriIter.next();
            while (metaIter.hasNext()) {
                Triple<Integer, RecordMetaDataLine, List<Writable>> t = metaIter.next();
                URI thisURI = t.getSecond().getURI();
                int nextLineIdx = t.getSecond().getLineNumber();

                //First: find the right URI for this record...
                while (!currentURI.equals(thisURI)) {
                    //Iterate to the next URI
                    currentURIIdx++;
                    if (currentURIIdx >= sortedURIs.size()) {
                        //Should never happen
                        throw new IllegalStateException(
                                        "Count not find URI " + thisURI + " in URIs list: " + sortedURIs);
                    }
                    currentURI = sortedURIs.get(currentURIIdx);
                    currentLineIdx = 0;
                    if (currentURI.equals(thisURI)) {
                        //Found the correct URI for this MetaData instance
                        closeIfRequired(currentUriIter);
                        currentUriIter = IOUtils.lineIterator(new InputStreamReader(currentURI.toURL().openStream()));
                        line = currentUriIter.next();
                    }
                }

                //Have the correct URI/iter open -> scan to the required line
                while (currentLineIdx < nextLineIdx && currentUriIter.hasNext()) {
                    line = currentUriIter.next();
                    currentLineIdx++;
                }
                if (currentLineIdx < nextLineIdx && !currentUriIter.hasNext()) {
                    throw new IllegalStateException("Could not get line " + nextLineIdx + " from URI " + currentURI
                                    + ": has only " + currentLineIdx + " lines");
                }
                t.setThird(Collections.<Writable>singletonList(new Text(line)));
            }
        } else {
            //Not URI based: String split, etc
            Iterator<String> iterator = getIterator(0);
            Iterator<Triple<Integer, RecordMetaDataLine, List<Writable>>> metaIter = list.iterator();
            int currentLineIdx = 0;
            String line = iterator.next();
            while (metaIter.hasNext()) {
                Triple<Integer, RecordMetaDataLine, List<Writable>> t = metaIter.next();
                int nextLineIdx = t.getSecond().getLineNumber();
                while (currentLineIdx < nextLineIdx && iterator.hasNext()) {
                    line = iterator.next();
                    currentLineIdx++;
                }
                t.setThird(Collections.<Writable>singletonList(new Text(line)));
            }
            closeIfRequired(iterator);
        }


        //Now, sort by the original (request) order:
        Collections.sort(list, new Comparator<Triple<Integer, RecordMetaDataLine, List<Writable>>>() {
            @Override
            public int compare(Triple<Integer, RecordMetaDataLine, List<Writable>> o1,
                            Triple<Integer, RecordMetaDataLine, List<Writable>> o2) {
                return Integer.compare(o1.getFirst(), o2.getFirst());
            }
        });

        //And return...
        List<Record> out = new ArrayList<>();
        for (Triple<Integer, RecordMetaDataLine, List<Writable>> t : list) {
            out.add(new org.datavec.api.records.impl.Record(t.getThird(), t.getSecond()));
        }
        return out;
    }
}
