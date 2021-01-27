/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.jdbc.driverfinder;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JClassLoading;

import java.io.IOException;
import java.io.InputStream;
import java.sql.Driver;
import java.util.HashSet;
import java.util.Objects;
import java.util.Properties;
import java.util.ServiceLoader;
import java.util.Set;

/**
 * JDBC Driver finder
 *
 * @author Adam Gibson
 */
@Slf4j
public class DriverFinder {

    public final static String ND4j_JDBC_PROPERTIES = "nd4j.jdbc.properties";
    public final static String JDBC_KEY = "jdbc.driver";
    private static Class<? extends Driver> clazz;
    private static Driver driver;

    public static Driver getDriver() {
        if (driver == null) {
            if (clazz == null)
                discoverDriverClazz();
            try {
                driver = clazz.newInstance();
            } catch (InstantiationException | IllegalAccessException e) {
                log.error("",e);
            }
        }
        return driver;
    }

    private static void discoverDriverClazz() {
        //All JDBC4 compliant drivers support ServiceLoader mechanism for discovery - https://stackoverflow.com/a/18297412
        ServiceLoader<Driver> drivers = ND4JClassLoading.loadService(Driver.class);
        Set<Class<? extends Driver>> driverClasses = new HashSet<>();
        for(Driver driver : drivers){
            driverClasses.add(driver.getClass());
        }

        if(driverClasses.isEmpty()){
            throw new IllegalStateException("No org.nd4j.jdbc drivers found on classpath via ServiceLoader");
        }

        if(driverClasses.size() != 1) {
            InputStream i = DriverFinder.class.getResourceAsStream("/" + ND4j_JDBC_PROPERTIES);
            if (i == null)
                throw new IllegalStateException("Only one jdbc driver allowed on the class path");
            else {
                Properties props = new Properties();
                try {
                    props.load(i);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

                String jdbcKeyClassName = props.getProperty(JDBC_KEY);
                Objects.requireNonNull(jdbcKeyClassName, "Unable to find jdbc driver. Please specify a "
                        + ND4j_JDBC_PROPERTIES + " with the key " + JDBC_KEY);

                DriverFinder.clazz = ND4JClassLoading.loadClassByName(jdbcKeyClassName);
                Objects.requireNonNull(DriverFinder.clazz, "Unable to find jdbc driver. Please specify a "
                        + ND4j_JDBC_PROPERTIES + " with the key " + JDBC_KEY);
            }
        }
    }
}
