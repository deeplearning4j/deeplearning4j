/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.nn.layers.mkldnn;

import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Method;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Base class for MLK-DNN Helpers
 * @author Alex Black
 */
public class BaseMKLDNNHelper {

    private static AtomicBoolean BACKEND_OK = null;
    private static AtomicBoolean FAILED_CHECK = null;

    public static boolean mklDnnEnabled(){
        if(BACKEND_OK == null){
            String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
            BACKEND_OK = new AtomicBoolean("CPU".equalsIgnoreCase(backend));
        }

        if(!BACKEND_OK.get() || (FAILED_CHECK != null && FAILED_CHECK.get())){
            //Wrong backend or already failed trying to check
            return false;
        }

        try{
            Class<?> c = Class.forName("org.nd4j.nativeblas.Nd4jCpu$Environment");
            Method m = c.getMethod("getInstance");
            Object instance = m.invoke(null);
            Method m2 = c.getMethod("isUseMKLDNN");
            boolean b = (Boolean)m2.invoke(instance);
            return b;
        } catch (Throwable t ){
            FAILED_CHECK = new AtomicBoolean(true);
            return false;
        }
    }

}
