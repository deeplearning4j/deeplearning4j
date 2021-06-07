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
import org.nd4j.linalg.factory.Nd4j;

import static org.deeplearning4j.common.config.DL4JSystemProperties.DISABLE_HELPER_PROPERTY;
import static org.deeplearning4j.common.config.DL4JSystemProperties.HELPER_DISABLE_DEFAULT_VALUE;

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
     * @param <T> the actual class type to be returned
     * @param cudnnHelperClassName the cudnn class name
     * @param oneDnnClassName the one dnn class name
     * @param layerHelperSuperClass the layer helper super class
     * @param layerName the name of the layer to be created
     * @param arguments the arguments to be used in creation of the layer
     * @return
     */
    public static <T extends LayerHelper> T createHelper(String cudnnHelperClassName,
                                                         String oneDnnClassName,
                                                         Class<? extends LayerHelper> layerHelperSuperClass,
                                                         String layerName,
                                                         Object... arguments) {

        Boolean disabled = Boolean.parseBoolean(System.getProperty(DISABLE_HELPER_PROPERTY,HELPER_DISABLE_DEFAULT_VALUE));
        if(disabled) {
            log.trace("Disabled helper creation, returning null");
            return null;
        }
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        LayerHelper helperRet = null;
        if("CUDA".equalsIgnoreCase(backend) && cudnnHelperClassName != null && !cudnnHelperClassName.isEmpty()) {
            if(DL4JClassLoading.loadClassByName(cudnnHelperClassName) != null) {
                log.debug("Attempting to initialize cudnn helper {}",cudnnHelperClassName);
                helperRet =  (LayerHelper) DL4JClassLoading.<LayerHelper>createNewInstance(
                        cudnnHelperClassName,
                        (Class<? super LayerHelper>) layerHelperSuperClass,
                        new Object[]{arguments});
                log.debug("Cudnn helper {} successfully initialized",cudnnHelperClassName);

            }
            else {
                log.warn("Unable to find class {}  using the classloader set for Dl4jClassLoading. Trying to use class loader that loaded the  class {} instead.",cudnnHelperClassName,layerHelperSuperClass.getName());
                ClassLoader classLoader = DL4JClassLoading.getDl4jClassloader();
                DL4JClassLoading.setDl4jClassloaderFromClass(layerHelperSuperClass);
                try {
                    helperRet =  (LayerHelper) DL4JClassLoading.<LayerHelper>createNewInstance(
                            cudnnHelperClassName,
                            (Class<? super LayerHelper>) layerHelperSuperClass,
                            arguments);

                } catch (Exception e) {
                    log.warn("Unable to use  helper implementation {} for helper type {}, please check your classpath. Falling back to built in  normal  methods for now.",cudnnHelperClassName,layerHelperSuperClass.getName());
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

        } else if("CPU".equalsIgnoreCase(backend) && oneDnnClassName != null && !oneDnnClassName.isEmpty()) {
            helperRet = DL4JClassLoading.<LayerHelper>createNewInstance(
                    oneDnnClassName,
                    arguments);
            log.trace("Created oneDNN helper: {}, layer {}", oneDnnClassName,layerName);
        }

        if (helperRet != null && !helperRet.checkSupported()) {
            log.debug("Removed helper {} as not supported", helperRet.getClass());
            return null;
        }

        return (T) helperRet;
    }

}
