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

//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#ifndef LIBCNPY_H_
#define LIBCNPY_H_


/**
 *
 */
#include <typeinfo>
#include <vector>
#include <cstdio>
#include <string>
#include <algorithm>
#include <map>
#include <assert.h>
#include <iostream>
#include <sstream>
#include<complex>
#include <cstring>
#include <algorithm>

#include <string>
#include <fstream>
#include <streambuf>
#include <op_boilerplate.h>
#include <dll.h>



namespace cnpy {

    /**
     * The numpy array
     */
    struct ND4J_EXPORT NpyArray {
        char* data;
        std::vector<unsigned int> shape;
        unsigned int wordSize;
        bool fortranOrder;
        void destruct() {
            delete[] data;
        }
    };

    struct ND4J_EXPORT npz_t : public std::map<std::string, NpyArray> {
        void destruct() {
            npz_t::iterator it = this->begin();
            for(; it != this->end(); ++it) (*it).second.destruct();
        }
    };

    /**
     *
     * @param path
     * @return
     */
    ND4J_EXPORT char* loadFile(const char *path);



    /**
     *
     * @return
     */
    char BigEndianTest();


    /**
     *
     * @param t
     * @return
     */
    char mapType(const std::type_info &t);

    /**
     *
     * @param T the type of the ndarray
     * @param data the data for the ndarray
     * @param shape the shape of the ndarray
     * @param ndims the rank of the ndarray
     * @return
     */
    template<typename T>
    std::vector<char> createNpyHeader(const T *data,
                                      const unsigned int *shape,
                                      const unsigned int ndims,
                                      unsigned int wordSize = 4);
    /**
     * Parse the numpy header from
     * the given file
     * based on the pointers passed in
     * @param fp the file to parse from
     * @param wordSize the size of
     * the individual elements
     * @param shape
     * @param ndims
     * @param fortranOrder
     */
    void parseNpyHeader(FILE *fp,
                        unsigned int &wordSize,
                        unsigned int *&shape,
                        unsigned int &ndims,
                        bool &fortranOrder);

    /**
    * Parse the numpy header from
    * the given file
    * based on the pointers passed in
    * @param header the file to parse from
    * @param word_size the size of
    * the individual elements
    * @param shape
    * @param ndims
    * @param fortran_order
    */
    void parseNpyHeaderPointer(
            const char *header,
            unsigned int& word_size,
            unsigned int*& shape,
            unsigned int& ndims,
            bool& fortran_order);
    /**
     *
     * @param fp
     * @param nrecs
     * @param global_header_size
     * @param global_header_offset
     */
    void parseZipFooter(FILE *fp,
                        unsigned short &nrecs,
                        unsigned int &global_header_size,
                        unsigned int &global_header_offset);

    /**
     *
     * @param fname
     * @return
     */
    npz_t npzLoad(std::string fname);

    /**
     *
     * @param fname
     * @param varname
     * @return
     */
    NpyArray npzLoad(std::string fname, std::string varname);

    /**
     *
     * @param fname
     * @return
     */
    NpyArray npyLoad(std::string fname);

    /**
    * Parse the numpy header from
    * the given file
    * based on the pointers passed in
    * @param fp the file to parse from
    * @param wordSize the size of
    * the individual elements
    * @param shape
    * @param ndims
    * @param fortranOrder
    */
    void parseNpyHeaderStr(std::string header,
                           unsigned int &wordSize,
                           unsigned int *&shape,
                           unsigned int &ndims,
                           bool &fortranOrder);


    /**
     *
     * @param fp
     * @return
     */
    int * shapeFromFile(FILE *fp);

    /**
     *
     * @param data
     * @return
     */
    int * shapeFromPointer(char *data);

    /**
     * Load the numpy array from the given file.
     * @param fp the file to load
     * @return the loaded array
     */
    ND4J_EXPORT NpyArray loadNpyFromFile(FILE *fp);

    /**
     *
     * @param data
     * @return
     */
    ND4J_EXPORT NpyArray loadNpyFromPointer(char *data);

    /**
   *
   * @param data
   * @return
   */
    ND4J_EXPORT NpyArray loadNpyFromHeader(char *data);


/**
* Parse the numpy header from
* the given file
* based on the pointers passed in
* @param fp the file to parse from
* @param word_size the size of
* the individual elements
* @param shape
* @param ndims
* @param fortran_order
*/
    void parseNpyHeader(std::string header,
                        unsigned int &word_size,
                        unsigned int *&shape,
                        unsigned int &ndims,
                        bool &fortran_order);

    /**
     *
     * @tparam T
     * @param i
     * @param pad
     * @param padval
     * @return
     */
    template<typename T>
    FORCEINLINE std::string tostring(T i, int pad = 0, char padval = ' ') {
        std::stringstream s;
        s << i;
        return s.str();
    }


    template<typename T>
    void npy_save(std::string fname, const T* data, const unsigned int* shape, const unsigned int ndims, std::string mode = "w");

}

/**
     *
     * @tparam T
     * @param lhs
     * @param rhs
     * @return
     */
template<typename T>
std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs);


#endif
