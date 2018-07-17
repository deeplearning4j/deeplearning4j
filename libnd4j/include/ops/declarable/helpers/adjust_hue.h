/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    static FORCEINLINE void rgb_to_hv(T r, T g, T b, T* h, T* v_min, T* v_max) {
        T v_mid;
        int h_category;
        // According to the figures in:
        // https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma
        // For the conditions, we don't care about the case where two components are
        // equal. It is okay to count it in either side in that case.
        if (r < g) {
            if (b < r) {
            // b < r < g
                *v_max = g;
                v_mid = r;
                *v_min = b;
                h_category = 1;
            } else if (b > g) {
            // r < g < b
                *v_max = b;
                v_mid = g;
                *v_min = r;
                h_category = 3;
            } else {
            // r < b < g
                *v_max = g;
                v_mid = b;
                *v_min = r;
                h_category = 2;
            }
        } else {
        // g < r
            if (b < g) {
            // b < g < r
                *v_max = r;
                v_mid = g;
                *v_min = b;
                h_category = 0;
            } else if (b > r) {
            // g < r < b
                *v_max = b;
                v_mid = r;
                *v_min = g;
                h_category = 4;
            } else {
            // g < b < r
                *v_max = r;
                v_mid = b;
                *v_min = g;
                h_category = 5;
            }
        }
        if (*v_max == *v_min) {
            *h = 0;
            return;
        }
        auto ratio = (v_mid - *v_min) / (*v_max - *v_min);
        bool increase = ((h_category & 0x1) == 0);
        *h = h_category + (increase ? ratio : (1 - ratio));
    }

    template <typename T>
    static FORCEINLINE void hv_to_rgb(T h, T v_min, T v_max, T* r, T* g, T* b) {
        int h_category = static_cast<int>(h);
        T ratio = h - (T)h_category;
        bool increase = ((h_category & 0x1) == 0);
        if (!increase)
            ratio = 1 - ratio;
        
        T v_mid = v_min + ratio * (v_max - v_min);
        // According to the figures in:
        // https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma
        switch (h_category) {
            case 0:
                *r = v_max;
                *g = v_mid;
                *b = v_min;
            break;
            case 1:
                *r = v_mid;
                *g = v_max;
                *b = v_min;
            break;
            case 2:
                *r = v_min;
                *g = v_max;
                *b = v_mid;
            break;
            case 3:
                *r = v_min;
                *g = v_mid;
                *b = v_max;
            break;
            case 4:
                *r = v_mid;
                *g = v_min;
                *b = v_max;
            break;
            case 5:
            default:
                *r = v_max;
                *g = v_min;
                *b = v_mid;
        }
    }

    template <typename T>
    void _adjust_hue(NDArray<T> *input, NDArray<T> *output, T delta, bool isNHWC);
}
}
}