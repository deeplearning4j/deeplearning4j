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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
//

#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


    void adjustHue(nd4j::LaunchContext* context, const NDArray *input, const NDArray* deltaScalarArr, NDArray *output, const int dimC);



////////////////////////////////////////////////////////////////////////////////
template <typename T>
FORCEINLINE _CUDA_HD void rgbToHsv(const T& r, const T& g, const T& b, T& h, T& s, T& v) {

    // h values are in range [0, 360)
    // s and v values are in range [0, 1]

    const T max = nd4j::math::nd4j_max<T>(r, nd4j::math::nd4j_max<T>(g, b));
    const T min = nd4j::math::nd4j_min<T>(r, nd4j::math::nd4j_min<T>(g, b));
    const T c  = max - min;
    const T _p6 = (T)1 / (T)6;
    // calculate h
    if(c == 0) {
        h = 0;
    }
    else if(max == r) {
        h = _p6 * ((g - b) / c) + (g >= b ? (T)0 : (T)1);
    }
    else if(max == g) {
        h = _p6 * ((b - r) / c + (T)2);
    }
    else { // max == b
        h = _p6 * ((r - g) / c + (T)4);
    }

    // calculate s
    s = max == (T)0 ? (T)0 : c / max;

    // calculate v
    v = max;// / 255.f;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
FORCEINLINE _CUDA_HD void hsvToRgb(const T& h, const T& s, const T& v, T& r, T& g, T& b) {

    const float sector = h * 6.f;
    const T c = v * s;

    if(0.f <= sector && sector < 1.f) {
        r = v;
        g = v - c * (1 - sector);
        b = v - c;
    }
    else if(1.f <= sector && sector < 2.f) {
        r = v - c * (sector - 1);
        g = v;
        b = v - c;
    }
    else if(2.f <= sector && sector < 3.f) {
        r = v - c;
        g = v;
        b = v - c * (3 - sector);
    }
    else if(3.f <= sector && sector < 4.f) {
        r = v - c;
        g = v - c * (sector - 3);
        b = v;
    }
    else if(4.f <= sector && sector < 5.f) {
        r = v - c * (5 - sector);
        g = v - c;
        b = v;
    }
    else {      // 5.f <= sector < 6.f
        r = v;
        g = v - c;
        b = v - c * (sector - 5);
    }

//    r *= 255;
//    g *= 255;
//    b *= 255;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
FORCEINLINE _CUDA_HD void rgbYuv(const T& r, const T& g, const T& b, T& y, T& u, T& v) {
    y =  static_cast<T>(0.299) * r + static_cast<T>(0.587) *g + static_cast<T>(0.114) * b;
    u = -static_cast<T>(0.14714119) * r - static_cast<T>(0.2888691) * g + static_cast<T>(0.43601035) * b;
    v = static_cast<T>(0.61497538) * r - static_cast<T>(0.51496512) * g - static_cast<T>(0.10001026) * b;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
FORCEINLINE _CUDA_HD void yuvRgb(const T& y, const T& u, const T& v, T& r, T& g, T& b) {
    r = y + static_cast<T>(1.13988303)  * v;
    g = y - static_cast<T>(0.394642334) * u - static_cast<T>(0.58062185) * v;
    b = y + static_cast<T>(2.03206185)  * u;
}

/*////////////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE _CUDA_HD void rgb_to_hv(T r, T g, T b, T* h, T* v_min, T* v_max) {
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

////////////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE _CUDA_HD void hv_to_rgb(T h, T v_min, T v_max, T* r, T* g, T* b) {
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

*/
}
}
}