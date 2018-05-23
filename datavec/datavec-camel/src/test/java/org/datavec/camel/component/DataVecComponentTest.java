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

package org.datavec.camel.component;

import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.component.mock.MockEndpoint;
import org.apache.camel.test.junit4.CamelTestSupport;
import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;

public class DataVecComponentTest extends CamelTestSupport {

    @ClassRule
    public static TemporaryFolder testDir = new TemporaryFolder();
    private static File dir;
    private static File irisFile;


    @BeforeClass
    public static void before() throws Exception {
        dir = testDir.newFolder();
        File iris = new ClassPathResource("iris.dat").getFile();
        irisFile = new File(dir, "iris.dat");
        FileUtils.copyFile(iris, irisFile );
    }



    @Test
    public void testDataVec() throws Exception {
        MockEndpoint mock = getMockEndpoint("mock:result");
        //1
        mock.expectedMessageCount(1);

        RecordReader reader = new CSVRecordReader();
        reader.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));
        Collection<Collection<Writable>> recordAssertion = new ArrayList<>();
        while (reader.hasNext())
            recordAssertion.add(reader.next());
        mock.expectedBodiesReceived(recordAssertion);
        assertMockEndpointsSatisfied();
    }

    @Override
    protected RouteBuilder createRouteBuilder() throws Exception {


        return new RouteBuilder() {
            public void configure() {
                from("file:" + dir.getAbsolutePath() + "?fileName=iris.dat&noop=true").unmarshal().csv()
                                .to("datavec://org.datavec.api.formats.input.impl.ListStringInputFormat?inputMarshaller=org.datavec.camel.component.ListStringInputMarshaller&writableConverter=org.datavec.api.io.converters.SelfWritableConverter")
                                .to("mock:result");
            }
        };
    }
}
