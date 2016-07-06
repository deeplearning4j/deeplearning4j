package org.canova.camel.component;

import lombok.Data;
import org.apache.camel.Consumer;
import org.apache.camel.Processor;
import org.apache.camel.Producer;
import org.apache.camel.impl.DefaultEndpoint;
import org.apache.camel.spi.Metadata;
import org.apache.camel.spi.UriEndpoint;
import org.apache.camel.spi.UriParam;
import org.apache.camel.spi.UriPath;
import org.canova.api.io.converters.SelfWritableConverter;

/**
 * Represents a canova endpoint.
 * @author Adam Gibson
 */
@UriEndpoint(scheme = "canova", title = "canova", syntax="canova:inputFormat/?outputFormat=?&inputMarshaller=?", consumerClass = CanovaConsumer.class,label = "canova")
@Data
public class CanovaEndpoint extends DefaultEndpoint {
    @UriPath @Metadata(required = "true")
    private String inputFormat;
    @UriParam(defaultValue = "")
    private String outputFormat;
    @UriParam @Metadata(required = "true")
    private String inputMarshaller;
    @UriParam(defaultValue = "org.canova.api.io.converters.SelfWritableConverter")
    private String writableConverter;

    public CanovaEndpoint(String uri, CanovaComponent component) {
        super(uri, component);
    }

    public CanovaEndpoint(String endpointUri) {
        super(endpointUri);
    }

    public Producer createProducer() throws Exception {
        return new CanovaProducer(this);
    }

    public Consumer createConsumer(Processor processor) throws Exception {
        return new CanovaConsumer(this, processor);
    }

    public boolean isSingleton() {
        return true;
    }

}
