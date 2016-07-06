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

package org.canova.api.records.writer.impl;


import org.canova.api.conf.Configuration;
import org.canova.api.io.data.Text;
import org.canova.api.records.writer.RecordWriter;
import org.canova.api.writable.Writable;

import java.io.*;
import java.nio.charset.Charset;
import java.util.Collection;

/**
 * Write to files.
 *
 * To set the path and configuration via configuration:
 * writeTo: org.canova.api.records.writer.path
 *
 * This is the path used to write to
 *
 *
 * @author Adam Gibson
 */
public  class FileRecordWriter implements RecordWriter {

    public static final Charset DEFAULT_CHARSET = Charset.forName("UTF-8");

    protected File writeTo;
    protected DataOutputStream out;
    public final static String NEW_LINE = "\n";
    private boolean append;
    public final static String PATH = "org.canova.api.records.writer.path";

    protected Charset encoding = DEFAULT_CHARSET;

    protected Configuration conf;

    public FileRecordWriter() {
    }

    public FileRecordWriter(File path) throws FileNotFoundException {
        this(path,false,DEFAULT_CHARSET);
    }


    public FileRecordWriter(File path,boolean append) throws FileNotFoundException {
        this(path,append,DEFAULT_CHARSET);
    }

    public FileRecordWriter(File path, boolean append, Charset encoding) throws FileNotFoundException{
        this.writeTo = path;
        this.append = append;
        this.encoding = encoding;
        out = new DataOutputStream(new FileOutputStream(writeTo,append));
    }


    /**
     * Initialized based on configuration
     * Set the following attributes in the conf:
     *
     * @param conf the configuration to use
     * @throws FileNotFoundException
     */
    public FileRecordWriter(Configuration conf) throws FileNotFoundException {
        setConf(conf);
    }

    @Override
    public void write(Collection<Writable> record) throws IOException {
        if(!record.isEmpty()) {
            Text t = (Text) record.iterator().next();
            t.write(out);
        }
    }

    @Override
    public void close() {
        if(out != null) {
            try {
                out.flush();
                out.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
        this.writeTo = new File(conf.get(PATH,"input.txt"));
        append = conf.getBoolean(APPEND,true);
        try {
            out = new DataOutputStream(new FileOutputStream(writeTo,append));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }

    }

    @Override
    public Configuration getConf() {
        return conf;
    }
}
