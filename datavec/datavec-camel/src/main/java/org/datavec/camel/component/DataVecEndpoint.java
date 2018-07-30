/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.camel.component;

import lombok.Data;
import org.apache.camel.Consumer;
import org.apache.camel.Processor;
import org.apache.camel.Producer;
import org.apache.camel.impl.DefaultEndpoint;
import org.apache.camel.spi.Metadata;
import org.apache.camel.spi.UriEndpoint;
import org.apache.camel.spi.UriParam;
import org.apache.camel.spi.UriPath;

/**
 * Represents a DataVec endpoint.
 * @author Adam Gibson
 */
@UriEndpoint(scheme = "datavec", title = "datavec", syntax = "datavec:inputFormat/?outputFormat=?&inputMarshaller=?",
                consumerClass = DataVecConsumer.class, label = "datavec")
@Data
public class DataVecEndpoint extends DefaultEndpoint {
    @UriPath
    @Metadata(required = "true")
    private String inputFormat;
    @UriParam(defaultValue = "")
    private String outputFormat;
    @UriParam
    @Metadata(required = "true")
    private String inputMarshaller;
    @UriParam(defaultValue = "org.datavec.api.io.converters.SelfWritableConverter")
    private String writableConverter;

    public DataVecEndpoint(String uri, DataVecComponent component) {
        super(uri, component);
    }

    public DataVecEndpoint(String endpointUri) {
        super(endpointUri);
    }

    public Producer createProducer() throws Exception {
        return new DataVecProducer(this);
    }

    public Consumer createConsumer(Processor processor) throws Exception {
        return new DataVecConsumer(this, processor);
    }

    public boolean isSingleton() {
        return true;
    }

}
