/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.text.sentenceiterator;

import lombok.NonNull;

import java.io.*;
import java.util.Iterator;

/**
 * Primitive single-line iterator, without any options involved.
 * Can be used over InputStream or File.
 *
 * Please note: for reset functionality, mark/reset should be supported by underlying InputStream.
 *
 * @author raver119@gmail.com
  */
public class BasicLineIterator implements SentenceIterator, Iterable<String> {

    private BufferedReader reader;
    private InputStream backendStream;
    private SentencePreProcessor preProcessor;
    private boolean internal = false;

    public BasicLineIterator(@NonNull File file) throws FileNotFoundException {
        this(new FileInputStream(file));
        this.internal = true;
    }

    public BasicLineIterator(@NonNull InputStream stream) {
        this.backendStream = stream;
        reader = new BufferedReader(new InputStreamReader(new BufferedInputStream(backendStream, 10 * 1024 * 1024)));
    }

    public BasicLineIterator(@NonNull String filePath) throws FileNotFoundException {
        this(new FileInputStream(filePath));
        this.internal = true;
    }

    @Override
    public synchronized String nextSentence() {
        try {
            return (preProcessor != null) ? this.preProcessor.preProcess(reader.readLine()) : reader.readLine();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public synchronized boolean hasNext() {
        try {
            return reader.ready();
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public synchronized void reset() {
        try {
            if (backendStream instanceof FileInputStream) {
                ((FileInputStream) backendStream).getChannel().position(0);
            } else
                backendStream.reset();
            reader = new BufferedReader(new InputStreamReader(new BufferedInputStream(backendStream, 10 * 10 * 1024)));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void finish() {
        try {
            if (this.internal && backendStream != null)
                backendStream.close();
            if (reader != null)
                reader.close();
        } catch (Exception e) {
            // do nothing here
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

    @Override
    protected void finalize() throws Throwable {
        try {
            if (this.internal && backendStream != null)
                backendStream.close();
            if (reader != null)
                reader.close();
        } catch (Exception e) {
            // do nothing here
            e.printStackTrace();
        }
        super.finalize();
    }

    /**
     * Implentation for Iterable interface.
     * Please note: each call for iterator() resets underlying SentenceIterator to the beginning;
     *
     * @return
     */
    @Override
    public Iterator<String> iterator() {
        this.reset();
        Iterator<String> ret = new Iterator<String>() {
            @Override
            public boolean hasNext() {
                return BasicLineIterator.this.hasNext();
            }

            @Override
            public String next() {
                return BasicLineIterator.this.nextSentence();
            }

            @Override
            public void remove() {
                throw new UnsupportedOperationException();
            }
        };

        return ret;
    }
}
