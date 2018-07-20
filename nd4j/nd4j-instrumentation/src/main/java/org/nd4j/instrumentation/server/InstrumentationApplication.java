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

package org.nd4j.instrumentation.server;

import io.dropwizard.Application;
import io.dropwizard.setup.Environment;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

/**
 * The instrumentation application
 *
 * @author Adam Gibson
 */
public class InstrumentationApplication extends Application<Nd4jInstrumentationConfiguration> {

    private String resourcePath = "org/nd4j/instrumentation/dropwizard.yml";
    private Environment env;

    public InstrumentationApplication(String resourcePath) {
        this.resourcePath = resourcePath;
    }

    public InstrumentationApplication() {}

    @Override
    public void run(Nd4jInstrumentationConfiguration nd4jInstrumentationConfiguration, Environment environment)
                    throws Exception {
        environment.jersey().register(new InstrumentationResource());
        this.env = environment;
    }

    /**
     * Start the server
     */
    public void start() {
        try {
            InputStream is = new ClassPathResource(resourcePath, InstrumentationApplication.class.getClassLoader())
                            .getInputStream();
            File tmpConfig = new File(resourcePath);
            if (!tmpConfig.getParentFile().exists())
                tmpConfig.getParentFile().mkdirs();
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(tmpConfig));
            IOUtils.copy(is, bos);
            bos.flush();
            run(new String[] {"server", tmpConfig.getAbsolutePath()});
            tmpConfig.deleteOnExit();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Stop the server
     * @throws Exception
     */
    public void stop() throws Exception {
        env.getAdminContext().stop();
    }


}
