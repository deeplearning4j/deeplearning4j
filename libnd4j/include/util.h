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

/* 
 * File:   util.h
 * Author: saudet
 *
 * Created on July 18, 2016, 1:28 PM
 */

#ifndef NATIVEOPERATIONS_UTIL_H
#define NATIVEOPERATIONS_UTIL_H

#ifdef WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include "pointercast.h"

static inline Nd4jLong microTime() {
#ifdef WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (Nd4jLong)count.QuadPart/freq.QuadPart;
#else
    timeval tv;
    gettimeofday(&tv, NULL);
    return (Nd4jLong)tv.tv_sec*1000000 + tv.tv_usec;
#endif
}

#endif /* NATIVEOPERATIONS_UTIL_H */
