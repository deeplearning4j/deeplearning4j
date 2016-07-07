package org.datavec.camel.component;

import java.util.Map;

import org.apache.camel.CamelContext;
import org.apache.camel.Endpoint;

import org.apache.camel.impl.UriEndpointComponent;

/**
 * Represents the component that manages {@link DataVecEndpoint}.
 */
public class DataVecComponent extends UriEndpointComponent {
    
    public DataVecComponent() {
        super(DataVecEndpoint.class);
    }

    public DataVecComponent(CamelContext context) {
        super(context, DataVecEndpoint.class);
    }

    @Override
    protected Endpoint createEndpoint(String uri, String remaining, Map<String, Object> parameters) throws Exception {
        DataVecEndpoint endpoint = new DataVecEndpoint(uri, this);
        setProperties(endpoint, parameters);
        endpoint.setInputFormat(remaining);
        return endpoint;
    }
}
