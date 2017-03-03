/*-
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.spark.util;

import org.apache.commons.io.IOUtils;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.spark.BaseSparkTest;
import org.datavec.spark.transform.utils.SparkUtils;
import org.junit.Test;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 22/12/2016.
 */
public class TestSparkUtil extends BaseSparkTest {

    @Test
    public void testWriteWritablesToFile() throws Exception {

        List<List<Writable>> l = new ArrayList<>();
        l.add(Arrays.<Writable>asList(new Text("abc"), new DoubleWritable(2.0), new IntWritable(-1)));
        l.add(Arrays.<Writable>asList(new Text("def"), new DoubleWritable(4.0), new IntWritable(-2)));

        File f = File.createTempFile("testSparkUtil", "txt");
        f.deleteOnExit();

        SparkUtils.writeWritablesToFile(f.getAbsolutePath(), ",", l, sc);

        List<String> lines = IOUtils.readLines(new FileInputStream(f));
        List<String> expected = Arrays.asList("abc,2.0,-1", "def,4.0,-2");

        assertEquals(expected, lines);

    }

}
