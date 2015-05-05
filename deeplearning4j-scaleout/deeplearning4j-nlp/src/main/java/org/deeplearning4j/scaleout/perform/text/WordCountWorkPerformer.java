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

package org.deeplearning4j.scaleout.perform.text;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

/**
 * Word count word performer
 * @author Adam Gibson
 */
public class WordCountWorkPerformer implements WorkerPerformer {

    private transient TokenizerFactory tokenizerFactory;
    public final static String TOKENIZER_CLASS = "org.deeplearning4j.scaleout.perform.text.tokenizerfactoryclass";


    @Override
    public void perform(Job job) {
        String sentence = (String) job.getWork();
        Counter<String> result = new Counter<>();
        Tokenizer tokenizer = tokenizerFactory.create(sentence);
        while(tokenizer.hasMoreTokens())
            result.incrementCount(tokenizer.nextToken(),1.0);
        job.setResult(result);

    }

    @Override
    public void update(Object... o) {

    }

    @Override
    public void setup(Configuration conf) {
        try {
            Class<? extends TokenizerFactory> clazz = (Class<? extends TokenizerFactory>) Class.forName(conf.get(TOKENIZER_CLASS, DefaultTokenizerFactory.class.getName()));
            tokenizerFactory = clazz.newInstance();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
