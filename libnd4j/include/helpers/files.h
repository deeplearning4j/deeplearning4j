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
// Methods to lookup files in $PATH
// adopted from http://stackoverflow.com/questions/2718915/check-if-file-exists-including-on-path
//

#ifndef LIBND4J_FILES_H
#define LIBND4J_FILES_H
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>


void *malloc_check(const char *what, size_t n);
char *strsave(const char *s, const char *lim);
char ** shellpath(void);
void freeshellpath (char *shellpath[]);
unsigned maxpathlen(char *path[], const char *base);
bool file_exists(char *name);

void *malloc_check(const char *what, size_t n) {
    void *p = malloc(n);
    if (p == NULL) {
        fprintf(stderr, "Cannot allocate %zu bytes to %s\n", n, what);
        exit(2);
    }
    return p;
}

char *strsave(const char *s, const char *lim) {
    if (lim == NULL)
        lim = s + strlen(s);
    char *p = (char *) malloc_check("save string", lim - s + 1);
    strncpy(p, s, lim-s);
    p[lim-s] = '\0';
    return p;
}

char ** shellpath(void) {
    const char *path = getenv("PATH");
    if (!path)
        path = "./";

    char **vector = // size is overkill
            (char **) malloc_check("hold path elements", strlen(path) * sizeof(*vector));
    const char *p = path;
    int next = 0;
    while (p) {
#ifdef _WIN32
        char *q = strchr(p, ';'); // windows uses ; as delimiter
#else
        char *q = strchr(p, ':'); // linux and derivatives use : as delimiter
#endif
        vector[next++] = strsave(p, q);
        p = q ? q + 1 : NULL;
    }
    vector[next] = NULL;
    return vector;
}

void freeshellpath (char *shellpath[]) {
    for (int i = 0; shellpath[i]; i++)
        free(shellpath[i]);
    free(shellpath);
}

unsigned maxpathlen(char *path[], const char *base) {
    unsigned blen = strlen(base);
    unsigned n = 0;
    for (int i = 0; path[i]; i++) {
        unsigned pn = strlen(path[i]);
        if (pn > n) n = pn;
    }
    return blen+n+1;
}
bool file_exists(char *name){
    printf("Trying file: [%s]\n", name);
    FILE *file;
    if (file = fopen(name, "r")) {
        fclose(file);
        return true;
    }
    return false;
}

bool checkFileInPath(const char *file) {
    char *path = getenv("PATH");
    char **listed = shellpath();
    size_t maxlen = maxpathlen(listed, file)+1;
    char *buf = (char *) malloc_check("hold path", maxlen);
    bool found = false;
    for (int i = 0; listed[i]; i++) {
        if (strlen(listed[i]) > 0) {
#ifdef _WIN32
            snprintf(buf, maxlen, "%s\\%s", listed[i], file);
#else
            snprintf(buf, maxlen, "%s/%s", listed[i], file);
#endif
            if (file_exists(buf)) {
                found = true;
                break;
            }
        }
    }
    free(buf);
    freeshellpath(listed);

    return found;
}


#endif //LIBND4J_FILES_H
