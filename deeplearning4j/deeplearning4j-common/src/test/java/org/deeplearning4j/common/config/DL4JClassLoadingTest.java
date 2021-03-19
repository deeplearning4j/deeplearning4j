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
package org.deeplearning4j.common.config;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import org.deeplearning4j.common.config.dummies.TestAbstract;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("Dl 4 J Class Loading Test")
class DL4JClassLoadingTest {

    private static final String PACKAGE_PREFIX = "org.deeplearning4j.common.config.dummies.";

    @Test
    @DisplayName("Test Create New Instance _ constructor Without Arguments")
    void testCreateNewInstance_constructorWithoutArguments() {
        /* Given */
        String className = PACKAGE_PREFIX + "TestDummy";
        /* When */
        Object instance = DL4JClassLoading.createNewInstance(className);
        /* Then */
        assertNotNull(instance);
        assertEquals(className, instance.getClass().getName());
    }

    @Test
    @DisplayName("Test Create New Instance _ constructor With Argument _ implicit Argument Types")
    void testCreateNewInstance_constructorWithArgument_implicitArgumentTypes() {
        /* Given */
        String className = PACKAGE_PREFIX + "TestColor";
        /* When */
        TestAbstract instance = DL4JClassLoading.createNewInstance(className, TestAbstract.class, "white");
        /* Then */
        assertNotNull(instance);
        assertEquals(className, instance.getClass().getName());
    }

    @Test
    @DisplayName("Test Create New Instance _ constructor With Argument _ explicit Argument Types")
    void testCreateNewInstance_constructorWithArgument_explicitArgumentTypes() {
        /* Given */
        String colorClassName = PACKAGE_PREFIX + "TestColor";
        String rectangleClassName = PACKAGE_PREFIX + "TestRectangle";
        /* When */
        TestAbstract color = DL4JClassLoading.createNewInstance(colorClassName, Object.class, new Class<?>[] { int.class, int.class, int.class }, 45, 175, 200);
        TestAbstract rectangle = DL4JClassLoading.createNewInstance(rectangleClassName, Object.class, new Class<?>[] { int.class, int.class, TestAbstract.class }, 10, 15, color);
        /* Then */
        assertNotNull(color);
        assertEquals(colorClassName, color.getClass().getName());
        assertNotNull(rectangle);
        assertEquals(rectangleClassName, rectangle.getClass().getName());
    }
}
