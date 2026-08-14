/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  * *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.nativeimage;

import org.junit.jupiter.api.Test;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;

public class Nd4jOperationRegistryBootstrapTest {

    @Test
    public void canonicalNd4jBootstrapBuildsTheOperationRegistryWithoutReentry() {
        assertNotNull(Nd4j.getExecutioner());

        DifferentialFunctionClassHolder registry =
                DifferentialFunctionClassHolder.getInstance();
        assertNotNull(registry);
        assertSame(registry, DifferentialFunctionClassHolder.getInstance());
        assertNotNull(DifferentialFunctionClassHolder.getInstance("add"));
    }
}
