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
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.io.InputStream;

/**
 * Created by agibsonccc on 1/29/15.
 */
public class TestConfigurableSentenceIterator extends ConfigurableSentenceIterator {


    public TestConfigurableSentenceIterator(Configuration conf) throws IOException {
        super(conf);
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
    public void reset() {

    }


}
