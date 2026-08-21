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

package org.eclipse.deeplearning4j.tokenizers.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.*;

/**
 * JavaCPP Presets for the HuggingFace tokenizers library.
 *
 * This class configures the JavaCPP bindings for the tokenizers Rust library
 * through its C wrapper layer. It follows the same pattern as nd4j's presets
 * to ensure compatibility and consistent behavior.
 *
 * The presets define:
 * - Header file mappings from the C wrapper
 * - Type mappings between C and Java
 * - Platform-specific compilation and linking settings
 * - Library loading and path configuration
 *
 * Adam Gibson
 */
@Properties(
        target = "org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative",
        helper = "org.eclipse.deeplearning4j.tokenizers.presets.TokenizersHelper",
        value = {
                @Platform(
                        define = {"TOKENIZERS_CPU"},
                        include = {
                                "tokenizers_c.h"
                        },
                        includepath = {
                                // JavaCPP-style include paths
                                "/org/eclipse/deeplearning4j/tokenizers/include/",
                                // Fallback to legacy paths for development
                                "../libtokenizers/target/native/org/eclipse/deeplearning4j/tokenizers/include/",
                                "../libtokenizers/include/"
                        },
                        compiler = {"cpp17", "nowarnings"},
                        library = "jnitokenizers",
                        link = "tokenizers_wrapper",
                        preload = "libtokenizers_wrapper"
                ),
                @Platform(
                        value = "linux",
                        preload = "gomp@.1",
                        preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/"}
                ),
                @Platform(
                        value = "linux-armhf",
                        preload = "gomp@.1",
                        preloadpath = {"/usr/arm-linux-gnueabihf/lib/", "/usr/lib/arm-linux-gnueabihf/"}
                ),
                @Platform(
                        value = "linux-arm64",
                        preload = "gomp@.1",
                        preloadpath = {"/usr/aarch64-linux-gnu/lib/", "/usr/lib/aarch64-linux-gnu/"}
                ),
                @Platform(
                        value = "windows",
                        preload = {
                                "libwinpthread-1",
                                "libgcc_s_seh-1",
                                "libstdc++-6",
                                "libtokenizers_wrapper"
                        }
                ),
                @Platform(
                        value = "macosx",
                        preload = "libtokenizers_wrapper"
                )
        }
)
public class TokenizersPresets implements InfoMapper, BuildEnabled {

    private Logger logger;
    private java.util.Properties properties;
    private String encoding;

    @Override
    public void init(Logger logger, java.util.Properties properties, String encoding) {
        this.logger = logger;
        this.properties = properties;
        this.encoding = encoding;
    }

    @Override
    public void map(InfoMap infoMap) {
        // Basic type mappings and annotations
        infoMap.put(new Info("TOKENIZERS_EXPORT", "TOKENIZERS_INLINE",
                        "TOKENIZERS_HOST", "TOKENIZERS_DEVICE", "TOKENIZERS_API")
                        .cppTypes().annotations())

                // Objectify the C API header to generate native methods
                .put(new Info("tokenizers_c.h").objectify())

                // Opaque pointer types - these map to the C API opaque handles
                .put(new Info("OpaqueTokenizer").pointerTypes("OpaqueTokenizer"))
                .put(new Info("OpaqueEncoding").pointerTypes("OpaqueEncoding"))
                .put(new Info("OpaqueDecodeStream").pointerTypes("OpaqueDecodeStream"))
                .put(new Info("OpaqueModelManager").pointerTypes("OpaqueModelManager"))

                // Enum mappings
                .put(new Info("TokenizerError").cast().valueTypes("int").pointerTypes("IntPointer"))
                .put(new Info("TokenizerResult").pointerTypes("TokenizerResult"))

                // String handling
                .put(new Info("char").valueTypes("byte").pointerTypes("@Cast(\"char*\") BytePointer",
                        "@Cast(\"char*\") String"))

                // Basic type mappings
                .put(new Info("bool").cast().valueTypes("boolean").pointerTypes("BooleanPointer", "boolean[]"))
                .put(new Info("size_t").cast().valueTypes("long").pointerTypes("SizeTPointer"))
                .put(new Info("uint32_t").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
                .put(new Info("int32_t").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
                .put(new Info("uint64_t").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))

                // Skip problematic functions
                .put(new Info("encode_batch").skip())
                .put(new Info("free_embedded_models").skip())

                // Function name mappings for better Java API names
                .put(new Info("create_tokenizer_from_file").javaNames("createTokenizerFromFile"))
                .put(new Info("create_tokenizer_from_json").javaNames("createTokenizerFromJson"))
                .put(new Info("free_tokenizer").javaNames("freeTokenizer"))
                .put(new Info("tokenizer_is_valid").javaNames("tokenizerIsValid"))
                .put(new Info("get_vocab_size").javaNames("getVocabSize"))
                .put(new Info("apply_chat_template").javaNames("applyChatTemplate"))
                .put(new Info("apply_chat_template_context").javaNames("applyChatTemplateContext"))
                .put(new Info("encode_text").javaNames("encodeText"))
                .put(new Info("free_encoding").javaNames("freeEncoding"))
                .put(new Info("encoding_get_length").javaNames("encodingGetLength"))
                .put(new Info("encoding_get_ids").javaNames("encodingGetIds"))
                .put(new Info("encoding_get_tokens").javaNames("encodingGetTokens"))
                .put(new Info("encoding_get_attention_mask").javaNames("encodingGetAttentionMask"))
                .put(new Info("encoding_get_type_ids").javaNames("encodingGetTypeIds"))
                .put(new Info("decode_ids").javaNames("decodeIds"))
                .put(new Info("create_decode_stream").javaNames("createDecodeStream"))
                .put(new Info("decode_stream_step").javaNames("decodeStreamStep"))
                .put(new Info("free_decode_stream").javaNames("freeDecodeStream"))
                .put(new Info("free_string").javaNames("freeString"))
                .put(new Info("get_token_id").javaNames("getTokenId"))
                .put(new Info("get_token").javaNames("getToken"))
                .put(new Info("create_model_manager").javaNames("createModelManager"))
                .put(new Info("free_model_manager").javaNames("freeModelManager"))
                .put(new Info("is_valid_model_file").javaNames("isValidModelFile"))
                .put(new Info("get_embedded_models").javaNames("getEmbeddedModels"))
                .put(new Info("free_encoding_batch").javaNames("freeEncodingBatch"))
                .put(new Info("encoding_get_offsets").javaNames("encodingGetOffsets"))
                .put(new Info("get_last_error").javaNames("getLastError"))
                .put(new Info("clear_last_error").javaNames("clearLastError"))
                .put(new Info("get_tokenizer_version").javaNames("getTokenizerVersion"))
                .put(new Info("get_build_info").javaNames("getBuildInfo"));

        // Platform-specific configurations
        String platform = properties.getProperty("platform", "");
        if (platform.contains("windows")) {
            infoMap.put(new Info("WINAPI", "__declspec(dllexport)").cppTypes().annotations());
        } else if (platform.contains("linux") || platform.contains("macosx")) {
            infoMap.put(new Info("__attribute__((visibility(\"default\")))").cppTypes().annotations());
        }

        // Skip C++ specific constructs
        infoMap.put(new Info("std::shared_ptr", "std::unique_ptr").skip())
                .put(new Info("std::initializer_list").skip());

        if (logger != null) {
            logger.info("TokenizersPresets: Configured mappings for platform: " + platform);
        }
    }
}
