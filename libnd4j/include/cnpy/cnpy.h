/*******************************************************************************
 * The MIT License
 *
 * Copyright (c) Carl Rogers, 2011
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
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
#include <array/DataType.h>


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
    ND4J_EXPORT char mapType(const std::type_info &t);

    template <typename T>
    ND4J_EXPORT char mapType();

    /**
     *
     * @param T the type of the ndarray
     * @param data the data for the ndarray
     * @param shape the shape of the ndarray
     * @param ndims the rank of the ndarray
     * @return
     */
    template<typename T>
    ND4J_EXPORT std::vector<char> createNpyHeader(const void *data,
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
    ND4J_EXPORT void parseNpyHeader(FILE *fp,
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
    ND4J_EXPORT void parseNpyHeaderPointer(
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
    ND4J_EXPORT void parseZipFooter(FILE *fp,
                        unsigned short &nrecs,
                        unsigned int &global_header_size,
                        unsigned int &global_header_offset);

    /**
     *
     * @param fname
     * @param varname
     * @return
     */
    ND4J_EXPORT NpyArray npzLoad(std::string fname, std::string varname);

    /**
     *
     * @param fname
     * @return
     */
    ND4J_EXPORT NpyArray npyLoad(std::string fname);

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
    ND4J_EXPORT void parseNpyHeaderStr(std::string header,
                           unsigned int &wordSize,
                           unsigned int *&shape,
                           unsigned int &ndims,
                           bool &fortranOrder);


    /**
     *
     * @param fp
     * @return
     */
    ND4J_EXPORT int* shapeFromFile(FILE *fp);

    /**
     *
     * @param data
     * @return
     */
    ND4J_EXPORT int* shapeFromPointer(char *data);

    /**
     * Load the numpy array from the given file.
     * @param fp the file to load
     * @return the loaded array
     */
    ND4J_EXPORT NpyArray loadNpyFromFile(FILE *fp);

    /**
     * Load the numpy array archive from the given file.
     * @param fp the file to load
     * @return the loaded archive
    */
    ND4J_EXPORT npz_t npzLoad(FILE* fp);
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


    ND4J_EXPORT npz_t npzLoad(std::string fname);

    ND4J_EXPORT nd4j::DataType dataTypeFromHeader(char *data);
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
    ND4J_EXPORT void parseNpyHeader(std::string header,
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
    ND4J_EXPORT void npy_save(std::string fname, const T* data, const unsigned int* shape, const unsigned int ndims, std::string mode = "w");

}

/**
     *
     * @tparam T
     * @param lhs
     * @param rhs
     * @return
     */
    template<typename T>
    ND4J_EXPORT std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs);


#endif
