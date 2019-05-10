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
// Created by raver119 on 11.10.2017.
//

#include "../MemoryUtils.h"
#include <helpers/logger.h>

#if defined(__APPLE__)
#include<mach/mach.h>
#include <sys/resource.h>
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)

#else
// linux
#include <sys/resource.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#endif


bool nd4j::memory::MemoryUtils::retrieveMemoryStatistics(nd4j::memory::MemoryReport &report) {
#if defined(__APPLE__)
    nd4j_debug("APPLE route\n", "");
/*
    struct task_basic_info t_info;
    mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

    if (KERN_SUCCESS != task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&t_info, &t_info_count))
        return false;

    report.setVM(t_info.resident_size);
    report.setRSS(t_info.resident_size);


    nd4j_debug("RSS: %lld; VM: %lld;\n", report.getRSS(), report.getVM());
*/
    struct rusage _usage;

    auto res = getrusage(RUSAGE_SELF, &_usage);

    report.setRSS(_usage.ru_maxrss);

    nd4j_debug("Usage: %lld; %lld; %lld; %lld;\n", _usage.ru_ixrss, _usage.ru_idrss, _usage.ru_isrss, _usage.ru_maxrss);

    return true;
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    nd4j_debug("WIN32 route\n", "");


#else
    nd4j_debug("LINUX route\n", "");
    int fd = open("/proc/self/statm", O_RDONLY, 0);
    if (fd >= 0) {
        char line[256];
        char* s;
        int n;
        lseek(fd, 0, SEEK_SET);
        if ((n = read(fd, line, sizeof(line))) > 0 && (s = (char*)memchr(line, ' ', n)) != NULL) {
            report.setRSS((Nd4jLong)(atoll(s + 1) * getpagesize()));
        }
        close(fd);
    }

    /*
    struct rusage _usage;

    auto res = getrusage(RUSAGE_SELF, &_usage);

    report.setRSS(_usage.ru_maxrss);

    //nd4j_printf("Usage: %lld; %lld; %lld; %lld;\n", _usage.ru_ixrss, _usage.ru_idrss, _usage.ru_isrss, _usage.ru_maxrss);
     */


    return true;
#endif

    return false;
}
