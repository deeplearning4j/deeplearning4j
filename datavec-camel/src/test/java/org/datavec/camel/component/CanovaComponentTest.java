package org.datavec.camel.component;

import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.component.mock.MockEndpoint;
import org.apache.camel.test.junit4.CamelTestSupport;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collection;

public class CanovaComponentTest extends CamelTestSupport {

    @Test
    public void testCanova() throws Exception {
        MockEndpoint mock = getMockEndpoint("mock:result");
        //1
        mock.expectedMessageCount(1);

        RecordReader reader = new CSVRecordReader();
        reader.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));
        Collection<Collection<Writable>> recordAssertion = new ArrayList<>();
        while(reader.hasNext())
            recordAssertion.add(reader.next());
        mock.expectedBodiesReceived(recordAssertion);
        assertMockEndpointsSatisfied();
    }

    @Override
    protected RouteBuilder createRouteBuilder() throws Exception {
        return new RouteBuilder() {
            public void configure() {
                from("file:src/test/resources/?fileName=iris.dat&noop=true")
                        .unmarshal().csv()
                        .to("canova://org.nd4j.etl4j.api.formats.input.impl.ListStringInputFormat?inputMarshaller=org.component.ListStringInputMarshaller&writableConverter=org.nd4j.etl4j.api.io.converters.SelfWritableConverter")
                        .to("mock:result");
            }
        };
    }
}
