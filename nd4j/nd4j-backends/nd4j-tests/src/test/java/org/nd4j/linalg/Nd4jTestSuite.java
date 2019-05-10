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

package org.nd4j.linalg;

import org.junit.runners.BlockJUnit4ClassRunner;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.ServiceLoader;

/**
 * Test suite for nd4j.
 *
 * This will run every combination of every unit test provided
 * that the backend's ordering and test line up.
 *
 * @author Adam Gibson
 */

public class Nd4jTestSuite extends BlockJUnit4ClassRunner {
    //the system property for what backends should run
    public final static String BACKENDS_TO_LOAD = "backends";
    private static List<Nd4jBackend> backends;
    static {
        ServiceLoader<Nd4jBackend> loadedBackends = ServiceLoader.load(Nd4jBackend.class);
        Iterator<Nd4jBackend> backendIterator = loadedBackends.iterator();
        backends = new ArrayList<>();
        while (backendIterator.hasNext())
            backends.add(backendIterator.next());


    }

    /**
     * Only called reflectively. Do not use programmatically.
     *
     * @param klass
     */
    public Nd4jTestSuite(Class<?> klass) throws Throwable {
        super(klass);
    }


    /**
     * Based on the jvm arguments, an empty list is returned
     * if all backends should be run.
     * If only certain backends should run, please
     * pass a csv to the jvm as follows:
     * -Dorg.nd4j.linalg.tests.backendstorun=your.class1,your.class2
     * @return the list of backends to run
     */
    public static List<String> backendsToRun() {
        List<String> ret = new ArrayList<>();
        String val = System.getProperty(BACKENDS_TO_LOAD, "");
        if (val.isEmpty())
            return ret;

        String[] clazzes = val.split(",");

        for (String s : clazzes)
            ret.add(s);
        return ret;

    }



}
