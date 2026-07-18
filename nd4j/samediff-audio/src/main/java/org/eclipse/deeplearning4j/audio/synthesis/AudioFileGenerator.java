/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */

package org.eclipse.deeplearning4j.audio.synthesis;

import java.nio.file.Path;

/**
 * A loaded audio model that materializes one completed output file.
 *
 * <p>The supplied directory is owned by the serving process. Implementations
 * must write the result inside that directory and return only after the file is
 * complete and closed.</p>
 */
@FunctionalInterface
public interface AudioFileGenerator extends AutoCloseable {

    GeneratedAudioFile generate(AudioSynthesisRequest request, Path outputDirectory) throws Exception;

    /**
     * Release any loaded model/tokenizer resources. Stateless generators need
     * no explicit cleanup and remain valid functional-interface lambdas.
     */
    @Override
    default void close() throws Exception {
        // No-op by default.
    }
}
