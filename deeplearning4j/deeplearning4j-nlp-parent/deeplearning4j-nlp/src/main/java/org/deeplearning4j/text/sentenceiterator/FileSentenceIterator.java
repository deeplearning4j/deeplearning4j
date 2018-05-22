/*-
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

package org.deeplearning4j.text.sentenceiterator;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Queue;
import java.util.zip.GZIPInputStream;

@SuppressWarnings("unchecked")
public class FileSentenceIterator extends BaseSentenceIterator {

    /*
     * Used as a pair for when
     * the number of sentences is not known
     */
    protected volatile Iterator<File> fileIterator;
    protected volatile Queue<String> cache;
    protected volatile LineIterator currLineIterator;
    protected volatile File file;
    protected volatile File currentFile;

    /**
     * Takes a single file or directory
     *
     * @param preProcessor the sentence pre processor
     * @param file         the file or folder to iterate over
     */
    public FileSentenceIterator(SentencePreProcessor preProcessor, File file) {
        super(preProcessor);
        this.file = file;
        cache = new java.util.concurrent.ConcurrentLinkedDeque<>();
        if (file.isDirectory())
            fileIterator = FileUtils.iterateFiles(file, null, true);
        else
            fileIterator = Arrays.asList(file).iterator();
    }

    public FileSentenceIterator(File dir) {
        this(null, dir);
    }


    @Override
    public String nextSentence() {
        String ret = null;
        if (!cache.isEmpty()) {
            ret = cache.poll();
            if (preProcessor != null)
                ret = preProcessor.preProcess(ret);
            return ret;
        } else {

            if (currLineIterator == null || !currLineIterator.hasNext())
                nextLineIter();

            for (int i = 0; i < 100000; i++) {
                if (currLineIterator != null && currLineIterator.hasNext()) {
                    String line = currLineIterator.nextLine();
                    if (line != null)
                        cache.add(line);
                    else
                        break;
                } else
                    break;
            }

            if (!cache.isEmpty()) {
                ret = cache.poll();
                if (preProcessor != null)
                    ret = preProcessor.preProcess(ret);
                return ret;
            }

        }


        if (!cache.isEmpty())
            ret = cache.poll();
        return ret;

    }


    private void nextLineIter() {
        if (fileIterator.hasNext()) {
            try {
                File next = fileIterator.next();
                currentFile = next;
                if (next.getAbsolutePath().endsWith(".gz")) {
                    if (currLineIterator != null)
                        currLineIterator.close();
                    currLineIterator = IOUtils.lineIterator(
                                    new BufferedInputStream(new GZIPInputStream(new FileInputStream(next))), "UTF-8");

                } else {
                    if (currLineIterator != null) {
                        currLineIterator.close();
                    }
                    currLineIterator = FileUtils.lineIterator(next);

                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public boolean hasNext() {
        return currLineIterator != null && currLineIterator.hasNext() || fileIterator.hasNext() || !cache.isEmpty();
    }


    @Override
    public void reset() {
        if (file.isFile())
            fileIterator = Arrays.asList(file).iterator();
        else
            fileIterator = FileUtils.iterateFiles(file, null, true);


    }


}
