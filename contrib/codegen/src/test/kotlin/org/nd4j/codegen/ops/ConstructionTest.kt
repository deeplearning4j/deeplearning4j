package org.nd4j.codegen.ops

import org.junit.jupiter.api.Test

/**
 * Test that each Namespace actually constructs properly.
 *
 * This is allows us to utilize run-time consistency checks during the build process - if tests are enabled.
 */
class ConstructionTest {

    @Test
    fun bitwise() { Bitwise() }

    @Test
    fun random() { Random() }

    @Test
    fun math() { Math() }

    @Test
    fun base() { SDBaseOps() }

    @Test
    fun loss() { SDLoss() }

    @Test
    fun cnn() { SDCNN() }

    @Test
    fun rnn() { SDRNN() }

    @Test
    fun image() { SDImage() }

    @Test
    fun nn() { NN() }
}