/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.nn.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.common.config.DL4JClassLoading;
import org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingHelper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Simple meta helper util class for instantiating
 * platform specific layer helpers that handle interaction with
 * lower level libraries like cudnn and onednn.
 *
 * @author Adam Gibson
 */
@Slf4j
public class HelperUtils {

    /**
     * Creates a {@link LayerHelper}
     * for use with platform specific code.
     * @param cudnnHelperClassName the cudnn class name
     * @param oneDnnClassName the one dnn class name
     * @param dataType the datatype to be used in the layer
     * @param layerHelperSuperClass the layer helper super class
     * @param layerName the name of the layer to be created
     * @param <T> the actual class type to be returned
     * @return
     */
    public static <T extends LayerHelper> T createHelper(String cudnnHelperClassName,
                                                         String oneDnnClassName,
                                                         DataType dataType,
                                                         Class<? extends LayerHelper> layerHelperSuperClass,
                                                         String layerName) {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        T helperRet = null;
        if("CUDA".equalsIgnoreCase(backend)) {
            if(DL4JClassLoading.loadClassByName(cudnnHelperClassName) != null) {
                helperRet =  DL4JClassLoading.createNewInstance(
                        cudnnHelperClassName,
                        layerHelperSuperClass,
                        dataType);
                log.debug("Cudnn helper {} successfully initialized",cudnnHelperClassName);

            }
            else {
                log.warn("Unable to find class {}  using the classloader set for Dl4jClassLoading. Trying to use class loader that loaded the  class {} instead.",cudnnHelperClassName,cudnnHelperClassName);
                ClassLoader classLoader = DL4JClassLoading.getDl4jClassloader();
                DL4JClassLoading.setDl4jClassloaderFromClass(layerHelperSuperClass);
                try {
                    return DL4JClassLoading.createNewInstance(
                            cudnnHelperClassName,
                            layerHelperSuperClass,
                            dataType);

                } catch (Exception e) {
                    log.warn("Unable to use  helper {}, please check your classpath. Falling back to built in  normal  methods for now.",cudnnHelperClassName);
                }

                log.warn("Returning class loader to original one.");
                DL4JClassLoading.setDl4jClassloader(classLoader);

            }

            if (helperRet != null && !helperRet.checkSupported()) {
                return null;
            }

            if(helperRet != null) {
                log.debug("{} successfully initialized",cudnnHelperClassName);
            }

        } else if("CPU".equalsIgnoreCase(backend)) {
            helperRet =  DL4JClassLoading.createNewInstance(
                    oneDnnClassName,
                    layerHelperSuperClass,
                    dataType);
            log.trace("Created oneDNN helper: {}, layer {}", oneDnnClassName,layerName);
        }

        if (helperRet != null && !helperRet.checkSupported()) {
            log.debug("Removed helper {} as not supported", helperRet.getClass());
            return null;
        }

        return helperRet;
    }

}
