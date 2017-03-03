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

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.cas.CAS;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.token.type.Sentence;
import org.cleartk.util.cr.FilesCollectionReader;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.annotator.TokenizerAnnotator;
import org.deeplearning4j.text.uima.UimaResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Iterates over and returns sentences
 * based on the passed in analysis engine
 * @author Adam Gibson
 *
 */
public class UimaSentenceIterator extends BaseSentenceIterator {

    protected volatile CollectionReader reader;
    protected volatile Iterator<String> sentences;
    protected String path;
    private static final Logger log = LoggerFactory.getLogger(UimaSentenceIterator.class);
    private static AnalysisEngine defaultAnalysisEngine;
    private UimaResource resource;


    public UimaSentenceIterator(SentencePreProcessor preProcessor, String path, UimaResource resource) {
        super(preProcessor);
        this.path = path;
        File f = new File(path);
        if (f.isFile()) {

            //more than a kilobyte break up the file (only do this for files


            try {

                this.reader = FilesCollectionReader.getCollectionReader(path);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }



        } else {
            try {
                this.reader = FilesCollectionReader.getCollectionReader(path);
            } catch (ResourceInitializationException e) {
                throw new RuntimeException(e);
            }
        }

        this.resource = resource;
    }


    public UimaSentenceIterator(SentencePreProcessor preProcessor, CollectionReader cr, UimaResource resource) {
        super(preProcessor);
        this.reader = cr;
        this.resource = resource;
    }


    public UimaSentenceIterator(String path, UimaResource resource) {
        this(null, path, resource);
    }

    @Override
    public synchronized String nextSentence() {
        if (sentences == null || !sentences.hasNext()) {
            try {
                if (getReader().hasNext()) {
                    CAS cas = resource.retrieve();

                    try {
                        getReader().getNext(cas);
                    } catch (Exception e) {
                        log.warn("Done iterating returning an empty string");
                        return "";
                    }


                    resource.getAnalysisEngine().process(cas);



                    List<String> list = new ArrayList<>();
                    for (Sentence sentence : JCasUtil.select(cas.getJCas(), Sentence.class)) {
                        list.add(sentence.getCoveredText());
                    }


                    sentences = list.iterator();
                    //needs to be next cas
                    while (!sentences.hasNext()) {
                        //sentence is empty; go to another cas
                        if (reader.hasNext()) {
                            cas.reset();
                            getReader().getNext(cas);
                            resource.getAnalysisEngine().process(cas);
                            for (Sentence sentence : JCasUtil.select(cas.getJCas(), Sentence.class)) {
                                list.add(sentence.getCoveredText());
                            }
                            sentences = list.iterator();
                        } else
                            return null;
                    }


                    String ret = sentences.next();
                    if (this.getPreProcessor() != null)
                        ret = this.getPreProcessor().preProcess(ret);
                    return ret;
                }

                return null;

            } catch (Exception e) {
                throw new RuntimeException(e);
            }

        } else {
            String ret = sentences.next();
            if (this.getPreProcessor() != null)
                ret = this.getPreProcessor().preProcess(ret);
            return ret;
        }



    }

    public UimaResource getResource() {
        return resource;
    }

    /**
     * Creates a uima sentence iterator with the given path
     * @param path the path to the root directory or file to read from
     * @return the uima sentence iterator for the given root dir or file
     * @throws Exception
     */
    public static SentenceIterator createWithPath(String path) throws Exception {
        return new UimaSentenceIterator(path,
                        new UimaResource(AnalysisEngineFactory.createEngine(AnalysisEngineFactory
                                        .createEngineDescription(TokenizerAnnotator.getDescription(),
                                                        SentenceAnnotator.getDescription()))));
    }


    @Override
    public synchronized boolean hasNext() {
        try {
            return getReader().hasNext() || sentences != null && sentences.hasNext();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    private synchronized CollectionReader getReader() {
        return reader;
    }


    @Override
    public void reset() {
        try {
            this.reader = FilesCollectionReader.getCollectionReader(path);
        } catch (ResourceInitializationException e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Return a sentence segmenter
     * @return a sentence segmenter
     */
    public static AnalysisEngine segmenter() {
        try {
            if (defaultAnalysisEngine == null)

                defaultAnalysisEngine = AnalysisEngineFactory.createEngine(
                                AnalysisEngineFactory.createEngineDescription(SentenceAnnotator.getDescription()));

            return defaultAnalysisEngine;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }



}
