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

import org.apache.uima.cas.CAS;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.token.type.Sentence;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.annotator.TokenizerAnnotator;
import org.deeplearning4j.text.uima.UimaResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Iterates over and returns sentences
 * based on the passed in analysis engine
 *
 * Database version of UimaSentenceIterator based off Adam Gibson's UimaSentenceIterator but extends BasicResultSetIterator
 *
 * Please note: for reset functionality, the underlying JDBC ResultSet must not be of TYPE_FORWARD_ONLY
 * To achieve this using postgres you can make your query using:
 * connection.prepareStatement(sql,ResultSet.TYPE_SCROLL_INSENSITIVE,ResultSet.CONCUR_READ_ONLY);
 *
 * @author Brad Heap nzv8fan@gmail.com
 */
public class UimaResultSetIterator extends BasicResultSetIterator {

    private UimaResource resource;
    protected volatile Iterator<String> sentences;
    private static final Logger log = LoggerFactory.getLogger(UimaSentenceIterator.class);

    /**
     * Constructor which builds a new UimaResource object
     * @param rs the database result set object to iterate over
     * @param columnName the name of the column containing text
     * @throws ResourceInitializationException
     */
    public UimaResultSetIterator(ResultSet rs, String columnName) throws ResourceInitializationException {
        this(rs, columnName,
                        new UimaResource(AnalysisEngineFactory.createEngine(AnalysisEngineFactory
                                        .createEngineDescription(TokenizerAnnotator.getDescription(),
                                                        SentenceAnnotator.getDescription()))));
    }

    /**
     * Constructor which takes an existing UimaResource object
     * @param rs the database result set object to iterate over
     * @param columnName the name of the column containing text
     * @param resource
     */
    public UimaResultSetIterator(ResultSet rs, String columnName, UimaResource resource) {
        super(rs, columnName);
        this.resource = resource;
    }

    @Override
    public synchronized String nextSentence() {

        if (sentences == null || !sentences.hasNext()) {
            // if we have no sentence get the next row from the database
            try {
                String text = super.nextSentence();

                if (text == null)
                    return "";

                CAS cas = resource.retrieve();
                cas.setDocumentText(text);
                //                log.info("Document text: " + text);

                resource.getAnalysisEngine().process(cas);

                List<String> list = new ArrayList<>();
                for (Sentence sentence : JCasUtil.select(cas.getJCas(), Sentence.class)) {
                    list.add(sentence.getCoveredText());
                }

                sentences = list.iterator();

                String ret = sentences.next();
                if (this.getPreProcessor() != null)
                    ret = this.getPreProcessor().preProcess(ret);
                //                    log.info("Sentence text: " + ret);
                return ret;

            } catch (Exception e) {
                throw new RuntimeException(e);
            }

        } else {
            String ret = sentences.next();
            if (this.getPreProcessor() != null)
                ret = this.getPreProcessor().preProcess(ret);
            //            log.info("Sentence text: " + ret);
            return ret;
        }
    }

    @Override
    public synchronized boolean hasNext() {
        try {
            if (sentences != null && sentences.hasNext())
                return true;
            return super.hasNext();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void reset() {
        sentences = null;
        super.reset();
    }

    @Override
    public void finish() {
        sentences = null;
        super.finish();
    }

}
