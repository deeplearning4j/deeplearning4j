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



namespace cnpy {

    /**
     * The numpy array
     */
    struct NpyArray {
        char* data;
        std::vector<unsigned int> shape;
        unsigned int wordSize;
        bool fortranOrder;
        void destruct() {
            delete[] data;
        }
    };

    struct npz_t : public std::map<std::string, NpyArray> {
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
    char* loadFile(char const *path);



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
                                      const unsigned
                                      int *shape,
                                      const unsigned int ndims,
                                      unsigned int wordSize);
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
    NpyArray loadNpyFromFile(FILE *fp);

    /**
     *
     * @param data
     * @return
     */
    NpyArray loadNpyFromPointer(char *data);

    /**
   *
   * @param data
   * @return
   */
    NpyArray loadNpyFromHeader(char *data);

    /**
     *
     * @tparam T
     * @param lhs
     * @param rhs
     * @return
     */
    template<typename T>
    std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs) {
        //write in little endian
        for(char byte = 0; byte < sizeof(T); byte++) {
            char val = *((char*)&rhs+byte);
            lhs.push_back(val);
        }

        return lhs;
    }

    /**
     *
     * @param lhs
     * @param rhs
     * @return
     */
    template<>
    std::vector<char>& operator+=(std::vector<char>& lhs,
                                  const std::string rhs);

    /**
     *
     * @param lhs
     * @param rhs
     * @return
     */
    template<>
    std::vector<char>& operator+=(std::vector<char>& lhs,
                                  const char* rhs);
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
    std::string tostring(T i, int pad = 0, char padval = ' ') {
        std::stringstream s;
        s << i;
        return s.str();
    }

    /**
     * Save the numpy array
     * @tparam T
     * @param fname the file
     * @param data the data for the ndarray
     * @param shape the shape of the ndarray
     * @param ndims the number of dimensions
     * for the ndarray
     * @param mode the mode for writing
     */
    template<typename T>
    void npy_save(std::string fname,
                  const T* data,
                  const unsigned int* shape,
                  const unsigned int ndims,
                  std::string mode = "w") {

        FILE* fp = NULL;

        if(mode == "a")
            fp = fopen(fname.c_str(),"r+b");

        if(fp) {
            //file exists. we need to append to it. read the header, modify the array size
            unsigned int word_size, tmp_dims;
            unsigned int* tmp_shape = 0;
            bool fortran_order;
            parseNpyHeader(fp,
                           word_size,
                           tmp_shape,
                           tmp_dims,
                           fortran_order);

            assert(!fortran_order);

            if(word_size != sizeof(T)) {
                std::cout<<"libnpy error: " << fname<< " has word size " << word_size<<" but npy_save appending data sized " << sizeof(T) <<"\n";
                assert( word_size == sizeof(T) );
            }

            if(tmp_dims != ndims) {
                std::cout<<"libnpy error: npy_save attempting to append misdimensioned data to "<<fname<<"\n";
                assert(tmp_dims == ndims);
            }

            for(int i = 1; i < ndims; i++) {
                if(shape[i] != tmp_shape[i]) {
                    std::cout<<"libnpy error: npy_save attempting to append misshaped data to " << fname << "\n";
                    assert(shape[i] == tmp_shape[i]);
                }
            }

            tmp_shape[0] += shape[0];

            fseek(fp,0,SEEK_SET);
            std::vector<char> header = createNpyHeader(data,tmp_shape,ndims);
            fwrite(&header[0],sizeof(char),header.size(),fp);
            fseek(fp,0,SEEK_END);

            delete[] tmp_shape;
        }
        else {
            fp = fopen(fname.c_str(),"wb");
            std::vector<char> header = createNpyHeader(data,shape,ndims);
            fwrite(&header[0],sizeof(char),header.size(),fp);
        }

        unsigned int nels = 1;
        for(int i = 0;i < ndims;i++) nels *= shape[i];

        fwrite(data,sizeof(T),nels,fp);
        fclose(fp);
    }


    /**
     *
     * @tparam T
     * @param data
     * @param shape
     * @param ndims
     * @return
     */
    template<typename T>
    std::vector<char> createNpyHeader(const T *data,
                                      const unsigned int *shape,
                                      const unsigned int ndims,
                                      unsigned int wordSize) {

        std::vector<char> dict;
        dict += "{'descr': '";
        dict += BigEndianTest();
        dict += mapType(typeid(T));
        dict += tostring(wordSize);
        dict += "', 'fortran_order': False, 'shape': (";
        dict += tostring(shape[0]);
        for(int i = 1; i < ndims;i++) {
            dict += ", ";
            dict += tostring(shape[i]);
        }

        if(ndims == 1)
            dict += ",";
        dict += "), }";
        //pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
        int remainder = 16 - (10 + dict.size()) % 16;
        dict.insert(dict.end(),remainder,' ');
        dict.back() = '\n';

        std::vector<char> header;
        header += (char) 0x93;
        header += "NUMPY";
        header += (char) 0x01; //major version of numpy format
        header += (char) 0x00; //minor version of numpy format
        header += (unsigned short) dict.size();
        header.insert(header.end(),dict.begin(),dict.end());
        std::vector<int> remove;
        for(int i = 0; i < header.size(); i++) {
            if(header[i] == '\0') {
                remove.push_back(i);
            }
        }

        for(int i = 0; i < remove.size(); i++) {
            header.erase(header.begin() + remove[i]);
        }

        return header;
    }


}

#endif
