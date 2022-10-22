/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.ops

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*

fun Random() = Namespace("Random") {
    val random = Mixin("random"){
        Arg(DATA_TYPE, "datatype"){ description = "Data type of the output variable"}
        Arg(LONG, "shape") { count = AtLeast(0); description = "Shape of the new random %INPUT_TYPE%, as a 1D array" }
        Output(NUMERIC, "output") { description = "Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution" }
    }

    val legacyRandom = Mixin("legacyRandom"){
        useMixin(random)
        javaPackage = "org.nd4j.linalg.api.ops.random.impl"
        legacy = true
    }

    val normalRandom = Mixin("normalRandom"){
        Arg(FLOATING_POINT, "mean") { description = "Mean value for the random array" }
        Arg(FLOATING_POINT, "stddev") { description = "Standard deviation for the random array" }
        useMixin(legacyRandom)
    }

    Op("bernoulli") {
        javaOpClass = "BernoulliDistribution"
        Arg(FLOATING_POINT, "p") { description = "Probability of value 1" }
        useMixin(legacyRandom)

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a Bernoulli distribution,
            with the specified probability. Array values will have value 1 with probability P and value 0 with probability
            1-P.
            """.trimIndent()
        }
    }

    Op("binomial") {
        javaOpClass = "BinomialDistribution"

        Arg(INT, "nTrials") { description = "Number of trials parameter for the binomial distribution" }
        Arg(FLOATING_POINT, "p") { description = "Probability of success for each trial" }
        useMixin(legacyRandom)

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a Binomial distribution,
            with the specified number of trials and probability.
            """.trimIndent()
        }
    }

    Op("exponential") {
        javaPackage = "org.nd4j.linalg.api.ops.random.custom"
        javaOpClass = "RandomExponential"

        val lambda = Arg(FLOATING_POINT, "lambda") { description = "lambda parameter" }
        Constraint("Must be positive") { lambda gt 0 }
        useMixin(random)


        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a exponential distribution:
            P(x) = lambda * exp(-lambda * x)
            """.trimIndent()
        }
    }

    Op("logNormal", normalRandom) {
        javaOpClass = "LogNormalDistribution"

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a Log Normal distribution,
            i.e., {@code log(x) ~ N(mean, stdev)}
            """.trimIndent()
        }
    }

    Op("normal", normalRandom) {
        javaPackage = "org.nd4j.linalg.api.ops.random.impl"
        javaOpClass = "GaussianDistribution"

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a Gaussian (normal) distribution,
            N(mean, stdev)<br>
            """.trimIndent()
        }
    }

    Op("normalTruncated", normalRandom) {
        javaOpClass = "TruncatedNormalDistribution"

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a Gaussian (normal) distribution,
            N(mean, stdev). However, any values more than 1 standard deviation from the mean are dropped and re-sampled
            """.trimIndent()
        }
    }

    Op("uniform") {
        javaOpClass = "UniformDistribution"

        Arg(FLOATING_POINT, "min") { description = "Minimum value" }
        Arg(FLOATING_POINT, "max") { description = "Maximum value." }
        useMixin(legacyRandom)

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a uniform distribution,
            U(min,max)
            """.trimIndent()
        }
    }
}