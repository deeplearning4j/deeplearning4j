/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.hadoop.nlp.text;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;


/**
 * Hdfs style sentence iterator. Provides base cases for iterating over files
 * and the baseline sentence iterator interface.
 *
 * @author Adam Gibson
 */
public abstract class ConfigurableSentenceIterator  implements SentenceIterator {
    protected Configuration conf;
    protected FileSystem fs;
    protected String rootPath;
    protected Path rootFilePath;
    protected Iterator<Path> paths;
    protected SentencePreProcessor preProcessor;
    public final static String ROOT_PATH = "org.depelearning4j.hadoop.nlp.rootPath";


    public ConfigurableSentenceIterator(Configuration conf) throws IOException {
        this.conf = conf;
        rootPath = conf.get(ROOT_PATH);
        fs = FileSystem.get(conf);
        if(rootPath == null)
            throw new IllegalArgumentException("Unable to create iterator from un specified file path");
        rootFilePath = new Path(rootPath);
        List<Path> paths = new ArrayList<>();
        find(paths,rootFilePath);
        this.paths = paths.iterator();
    }


    private void find(List<Path> paths,Path currentFile) throws IOException {
        if(fs.isDirectory(currentFile)) {
            FileStatus[] statuses = fs.listStatus(currentFile);
            for(FileStatus status : statuses)
                find(paths,status.getPath());

        }

        else
           paths.add(currentFile);

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
    public boolean hasNext() {
        return paths.hasNext();
    }

    @Override
    public void finish() {
        try {
            fs.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public SentencePreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public void setPreProcessor(SentencePreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

}
