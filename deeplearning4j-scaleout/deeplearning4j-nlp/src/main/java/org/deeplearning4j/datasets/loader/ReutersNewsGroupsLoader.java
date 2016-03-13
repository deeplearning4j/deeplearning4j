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

package org.deeplearning4j.datasets.loader;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.util.ArchiveUtils;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.bagofwords.vectorizer.LegacyBagOfWordsVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.LegacyTfidfVectorizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class ReutersNewsGroupsLoader extends BaseDataFetcher {

    private TextVectorizer textVectorizer;
    private boolean tfidf;
    public final static String NEWSGROUP_URL = "http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz";
    private File reutersRootDir;
    private static final Logger log = LoggerFactory.getLogger(ReutersNewsGroupsLoader.class);
    private DataSet load;


    public ReutersNewsGroupsLoader(boolean tfidf) throws Exception {
        getIfNotExists();
        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(reutersRootDir);
        List<String> labels =new ArrayList<>();
        for(File f : reutersRootDir.listFiles()) {
            if(f.isDirectory())
                labels.add(f.getName());
        }
        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();

        if(tfidf)
            this.textVectorizer = new LegacyTfidfVectorizer.Builder()
                    .iterate(iter).labels(labels).tokenize(tokenizerFactory).build();

        else
            this.textVectorizer = new LegacyBagOfWordsVectorizer.Builder()
                    .iterate(iter).labels(labels).tokenize(tokenizerFactory).build();

        load = this.textVectorizer.vectorize();
    }

    private void getIfNotExists() throws Exception {
        String home = System.getProperty("user.home");
        String rootDir = home + File.separator + "reuters";
        reutersRootDir = new File(rootDir);
        if(!reutersRootDir.exists())
            reutersRootDir.mkdir();
        else if(reutersRootDir.exists())
            return;


        File rootTarFile = new File(reutersRootDir,"20news-18828.tar.gz");
        if(rootTarFile.exists())
            rootTarFile.delete();
        rootTarFile.createNewFile();

        FileUtils.copyURLToFile(new URL(NEWSGROUP_URL), rootTarFile);
        ArchiveUtils.unzipFileTo(rootTarFile.getAbsolutePath(), reutersRootDir.getAbsolutePath());
        rootTarFile.delete();
        FileUtils.copyDirectory(new File(reutersRootDir,"20news-18828"),reutersRootDir);
        FileUtils.deleteDirectory(new File(reutersRootDir,"20news-18828"));
        if(reutersRootDir.listFiles() == null)
            throw new IllegalStateException("No files found!");

    }


    /**
     * Fetches the next dataset. You need to call this
     * to getFromOrigin a new dataset, otherwise {@link #next()}
     * just returns the last data applyTransformToDestination fetch
     *
     * @param numExamples the number of examples to fetch
     */
    @Override
    public void fetch(int numExamples) {
        List<DataSet> newData = new ArrayList<>();
        for(int grabbed = 0; grabbed < numExamples && cursor < load.numExamples(); cursor++,grabbed++) {
            newData.add(load.get(cursor));
        }

        this.curr = DataSet.merge(newData);

    }
}
