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
package org.nd4j.samediff.frameworkimport.onnx.export

/**
 * Configuration options for exporting SameDiff models to ONNX format.
 *
 * @property opsetVersion The ONNX opset version to target (default: 13)
 * @property irVersion The ONNX IR version (default: 7)
 * @property producerName Name of the producer (default: "nd4j-samediff")
 * @property producerVersion Version of the producer
 * @property domain The model domain (default: "")
 * @property modelVersion The model version (default: 1)
 * @property docString Optional documentation string for the model
 * @property includeTrainingState Whether to export training/optimizer state to a companion file
 * @property graphName Name for the graph (default: "main")
 *
 * @author Adam Gibson
 */
data class OnnxExportConfig(
    val opsetVersion: Long = 13L,
    val irVersion: Long = 7L,
    val producerName: String = "nd4j-samediff",
    val producerVersion: String = "1.0.0",
    val domain: String = "",
    val modelVersion: Long = 1L,
    val docString: String? = null,
    val includeTrainingState: Boolean = false,
    val graphName: String = "main"
) {
    companion object {
        /**
         * Default configuration for ONNX export.
         */
        @JvmStatic
        val DEFAULT = OnnxExportConfig()

        /**
         * Configuration for maximum compatibility with older runtimes.
         */
        @JvmStatic
        val COMPATIBLE = OnnxExportConfig(opsetVersion = 9L, irVersion = 4L)

        /**
         * Configuration with training state export enabled.
         */
        @JvmStatic
        val WITH_TRAINING = OnnxExportConfig(includeTrainingState = true)

        /**
         * Builder for creating custom configurations.
         */
        @JvmStatic
        fun builder(): Builder = Builder()
    }

    /**
     * Builder class for OnnxExportConfig.
     */
    class Builder {
        private var opsetVersion: Long = 13L
        private var irVersion: Long = 7L
        private var producerName: String = "nd4j-samediff"
        private var producerVersion: String = "1.0.0"
        private var domain: String = ""
        private var modelVersion: Long = 1L
        private var docString: String? = null
        private var includeTrainingState: Boolean = false
        private var graphName: String = "main"

        fun opsetVersion(version: Long): Builder {
            this.opsetVersion = version
            return this
        }

        fun irVersion(version: Long): Builder {
            this.irVersion = version
            return this
        }

        fun producerName(name: String): Builder {
            this.producerName = name
            return this
        }

        fun producerVersion(version: String): Builder {
            this.producerVersion = version
            return this
        }

        fun domain(domain: String): Builder {
            this.domain = domain
            return this
        }

        fun modelVersion(version: Long): Builder {
            this.modelVersion = version
            return this
        }

        fun docString(doc: String?): Builder {
            this.docString = doc
            return this
        }

        fun includeTrainingState(include: Boolean): Builder {
            this.includeTrainingState = include
            return this
        }

        fun graphName(name: String): Builder {
            this.graphName = name
            return this
        }

        fun build(): OnnxExportConfig = OnnxExportConfig(
            opsetVersion = opsetVersion,
            irVersion = irVersion,
            producerName = producerName,
            producerVersion = producerVersion,
            domain = domain,
            modelVersion = modelVersion,
            docString = docString,
            includeTrainingState = includeTrainingState,
            graphName = graphName
        )
    }
}
