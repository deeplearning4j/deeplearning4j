package org.component;

import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.component.mock.MockEndpoint;
import org.apache.camel.test.junit4.CamelTestSupport;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
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
                        .to("canova://org.canova.api.formats.input.impl.ListStringInputFormat?inputMarshaller=org.component.ListStringInputMarshaller&writableConverter=org.canova.api.io.converters.SelfWritableConverter")
                        .to("mock:result");
            }
        };
    }
}
