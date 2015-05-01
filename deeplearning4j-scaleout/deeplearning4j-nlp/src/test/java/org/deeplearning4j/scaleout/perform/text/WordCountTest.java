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

import static org.junit.Assert.*;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;
import org.junit.Test;

/**
 * Created by agibsonccc on 11/29/14.
 */
public class WordCountTest {
    @Test
    public void testWordCount() {
        WorkerPerformer performer = new WordCountWorkPerformer();
        Configuration conf = new Configuration();
        performer.setup(conf);
        Job j = new Job("This is one sentence.","");
        performer.perform(j);
        Counter<String> result = (Counter<String>) j.getResult();
        assertEquals(1.0,result.getCount("This"),1e-1);

    }



}
