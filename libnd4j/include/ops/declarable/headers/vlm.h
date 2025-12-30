/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Adam Gibson
//
// Vision Language Model (VLM) operations header
// These operations support multimodal AI models that combine vision and language.
//

#ifndef LIBND4J_HEADERS_VLM_H
#define LIBND4J_HEADERS_VLM_H

#include <ops/declarable/headers/common.h>

namespace sd {
namespace ops {

/**
 * vlm_vision_encode - Vision encoder for VLM models
 *
 * Encodes image features using a vision backbone (e.g., ViT, CLIP).
 * Uses GGML for accelerated computation when available.
 *
 * Input:
 *   0: input image tensor [batch, channels, height, width] or [batch, height, width, channels]
 *
 * Output:
 *   0: encoded vision features [batch, num_patches, embedding_dim]
 *
 * Integer arguments:
 *   0: embedding dimension (default: 768)
 *   1: number of attention heads (default: 12)
 *
 * Float arguments:
 *   0: layer norm epsilon (default: 1e-5)
 */
#if NOT_EXCLUDED(OP_vlm_vision_encode)
DECLARE_CUSTOM_OP(vlm_vision_encode, 1, 1, false, 0, 0);
#endif

/**
 * vlm_image_embed - Image embedding for VLM models
 *
 * Converts image patches to embeddings through linear projection.
 *
 * Input:
 *   0: input image tensor [batch, channels, height, width]
 *
 * Output:
 *   0: patch embeddings [batch, num_patches, embedding_dim]
 *
 * Integer arguments:
 *   0: patch size (default: 16)
 *   1: embedding dimension (default: 768)
 */
#if NOT_EXCLUDED(OP_vlm_image_embed)
DECLARE_CUSTOM_OP(vlm_image_embed, 1, 1, false, 0, 0);
#endif

/**
 * vlm_patch_embed - Patch extraction and embedding
 *
 * Extracts non-overlapping patches from images and projects them.
 *
 * Input:
 *   0: input image tensor [batch, channels, height, width]
 *
 * Output:
 *   0: patch embeddings [batch, num_patches, patch_dim]
 *
 * Integer arguments:
 *   0: patch size (default: 16)
 *   1: stride (default: patch_size, for non-overlapping)
 *   2: include CLS token (0=no, 1=yes, default: 1)
 */
#if NOT_EXCLUDED(OP_vlm_patch_embed)
DECLARE_CUSTOM_OP(vlm_patch_embed, 1, 1, false, 0, 0);
#endif

/**
 * vlm_cross_attention - Cross attention between vision and language
 *
 * Implements cross-attention mechanism for multimodal fusion.
 *
 * Input:
 *   0: query tensor (language) [batch, seq_len, dim]
 *   1: key tensor (vision) [batch, num_patches, dim]
 *   2: value tensor (vision) [batch, num_patches, dim]
 *
 * Output:
 *   0: attention output [batch, seq_len, dim]
 *
 * Integer arguments:
 *   0: number of attention heads (default: 8)
 *   1: causal mask (0=no, 1=yes, default: 0)
 *
 * Float arguments:
 *   0: attention scale (default: 1/sqrt(head_dim))
 */
#if NOT_EXCLUDED(OP_vlm_cross_attention)
DECLARE_CUSTOM_OP(vlm_cross_attention, 3, 1, false, 0, 0);
#endif

/**
 * vlm_multimodal_fusion - Fusion of vision and language representations
 *
 * Combines vision and language features using various fusion strategies.
 *
 * Input:
 *   0: vision features [batch, num_patches, dim]
 *   1: language features [batch, seq_len, dim]
 *
 * Output:
 *   0: fused features [batch, combined_len, dim]
 *
 * Integer arguments:
 *   0: fusion type (0=concatenate, 1=add, 2=multiply, default: 0)
 */
#if NOT_EXCLUDED(OP_vlm_multimodal_fusion)
DECLARE_CUSTOM_OP(vlm_multimodal_fusion, 2, 1, false, 0, 0);
#endif

/**
 * vlm_vision_projection - Project vision features to language space
 *
 * Linear projection to align vision features with language model dimensions.
 *
 * Input:
 *   0: vision features [batch, num_patches, vision_dim]
 *   1: projection weights [vision_dim, language_dim]
 *
 * Output:
 *   0: projected features [batch, num_patches, language_dim]
 */
#if NOT_EXCLUDED(OP_vlm_vision_projection)
DECLARE_CUSTOM_OP(vlm_vision_projection, 2, 1, false, 0, 0);
#endif

/**
 * vlm_image_preprocess - Image preprocessing for VLM
 *
 * Normalizes and prepares images for VLM input.
 * Applies mean subtraction and standard deviation normalization.
 *
 * Input:
 *   0: input image [batch, channels, height, width] or [batch, height, width, channels]
 *
 * Output:
 *   0: preprocessed image [batch, channels, height, width]
 *
 * Float arguments:
 *   0-2: mean values for R, G, B (default: ImageNet values 0.485, 0.456, 0.406)
 *   3-5: std values for R, G, B (default: ImageNet values 0.229, 0.224, 0.225)
 */
#if NOT_EXCLUDED(OP_vlm_image_preprocess)
DECLARE_CUSTOM_OP(vlm_image_preprocess, 1, 1, false, 0, 0);
#endif

/**
 * vlm_2d_position_encode - 2D sinusoidal position encoding for patches
 *
 * Adds 2D position information to patch embeddings using sinusoidal encoding.
 *
 * Input:
 *   0: patch embeddings [batch, num_patches, dim]
 *
 * Output:
 *   0: position-encoded patches [batch, num_patches, dim]
 *
 * Integer arguments:
 *   0: grid height (default: 14)
 *   1: grid width (default: 14)
 *
 * Float arguments:
 *   0: temperature for sinusoidal encoding (default: 10000.0)
 */
#if NOT_EXCLUDED(OP_vlm_2d_position_encode)
DECLARE_CUSTOM_OP(vlm_2d_position_encode, 1, 1, false, 0, 0);
#endif

}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HEADERS_VLM_H
