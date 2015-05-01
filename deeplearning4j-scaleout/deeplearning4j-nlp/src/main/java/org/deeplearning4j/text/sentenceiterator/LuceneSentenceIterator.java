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

package org.deeplearning4j.text.sentenceiterator;

import org.apache.commons.compress.utils.IOUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.Bits;
import org.deeplearning4j.berkeley.StringUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Lucene sentence iterator.
 * Read a sentence index
 * and produce sentences.
 *
 * @author Adam Gibson
 */
public class LuceneSentenceIterator implements SentenceIterator {

    private Directory dir;
    private transient IndexReader reader;
    public final static String WORD_FIELD = "word";
    private int index = 0;
    private SentencePreProcessor preProcessor;
    private List<Integer> docs;

    public LuceneSentenceIterator(Directory dir) {
        try {
            this.dir = dir;
            reader = DirectoryReader.open(dir);
            docs = allDocs();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public String nextSentence() {
        Document doc = null;
        try {
            doc = reader.document(docs.get(index));
            index++;
        } catch (IOException e) {
            e.printStackTrace();
        }

        String[] values = doc.getValues(WORD_FIELD);
        return StringUtils.join(values," ");
    }

    @Override
    public boolean hasNext() {
        return index < docs.size();
    }

    @Override
    public void reset() {
        index = 0;
    }

    @Override
    public void finish() {
        IOUtils.closeQuietly(reader);
    }

    @Override
    public SentencePreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public void setPreProcessor(SentencePreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }


    private List<Integer> allDocs() {
        List<Integer> docIds = new ArrayList<>();
        Bits liveDocs = MultiFields.getLiveDocs(reader);
        for(int i = 0; i < reader.maxDoc() + 1; i++) {
            if (liveDocs != null && !liveDocs.get(i))
                continue;

            docIds.add(i);
        }
        return docIds;
    }


}
