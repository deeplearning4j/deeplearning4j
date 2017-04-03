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
#include <zlib.h>
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
                                      const unsigned int ndims);
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
     * Save a numpy archive
     * @tparam T the
     * @param zipname the file zip name
     * @param fname the file name
     * @param data the data for the array
     * @param shape the shape for the ndarray
     * @param ndims the number of dimensions for the array
     * @param mode the mode for writing
     */
    template<typename T>
    void npzSave(std::string zipname,
                 std::string fname,
                 const T *data,
                 const unsigned int *shape,
                 const unsigned int ndims,
                 std::string mode = "w") {
        //first, append a .npy to the fname
        fname += ".npy";

        //now, on with the show
        FILE* fp = NULL;
        unsigned short nrecs = 0;
        unsigned int global_header_offset = 0;
        std::vector<char> global_header;

        if(mode == "a") fp = fopen(zipname.c_str(),"r+b");

        if(fp) {
            //zip file exists. we need to add a new npy file to it.
            //first read the footer. this gives us the offset and size of the global header
            //then read and store the global header. 
            //below, we will write the the new data at the start of the global header then append the global header and footer below it
            unsigned int global_header_size;
            parseZipFooter(fp, nrecs, global_header_size, global_header_offset);
            fseek(fp,global_header_offset,SEEK_SET);
            global_header.resize(global_header_size);
            size_t res = fread(&global_header[0],sizeof(char),global_header_size,fp);
            if(res != global_header_size){
                throw std::runtime_error("npz_save: header read error while adding to existing zip");
            }

            fseek(fp,global_header_offset,SEEK_SET);
        }
        else {
            fp = fopen(zipname.c_str(),"wb");
        }

        std::vector<char> npy_header = createNpyHeader(data,shape,ndims);

        unsigned long nels = 1;
        for (int m = 0; m<ndims; m++ ) nels *= shape[m];
        int nbytes = nels*sizeof(T) + npy_header.size();

        //get the CRC of the data to be added
        unsigned int crc = crc32(0L,(unsigned char*)&npy_header[0],npy_header.size());
        crc = crc32(crc,(unsigned char*)data,nels*sizeof(T));

        //build the local header
        std::vector<char> local_header;
        local_header += "PK"; //first part of sig
        local_header += (unsigned short) 0x0403; //second part of sig
        local_header += (unsigned short) 20; //min version to extract
        local_header += (unsigned short) 0; //general purpose bit flag
        local_header += (unsigned short) 0; //compression method
        local_header += (unsigned short) 0; //file last mod time
        local_header += (unsigned short) 0;     //file last mod date
        local_header += (unsigned int) crc; //crc
        local_header += (unsigned int) nbytes; //compressed size
        local_header += (unsigned int) nbytes; //uncompressed size
        local_header += (unsigned short) fname.size(); //fname length
        local_header += (unsigned short) 0; //extra field length
        local_header += fname;

        //build global header
        global_header += "PK"; //first part of sig
        global_header += (unsigned short) 0x0201; //second part of sig
        global_header += (unsigned short) 20; //version made by
        global_header.insert(global_header.end(),local_header.begin()+4,local_header.begin()+30);
        global_header += (unsigned short) 0; //file comment length
        global_header += (unsigned short) 0; //disk number where file starts
        global_header += (unsigned short) 0; //internal file attributes
        global_header += (unsigned int) 0; //external file attributes
        global_header += (unsigned int) global_header_offset; //relative offset of local file header, since it begins where the global header used to begin
        global_header += fname;

        //build footer
        std::vector<char> footer;
        footer += "PK"; //first part of sig
        footer += (unsigned short) 0x0605; //second part of sig
        footer += (unsigned short) 0; //number of this disk
        footer += (unsigned short) 0; //disk where footer starts
        footer += (unsigned short) (nrecs + 1); //number of records on this disk
        footer += (unsigned short) (nrecs + 1); //total number of records
        footer += (unsigned int) global_header.size(); //nbytes of global headers
        footer += (unsigned int) (global_header_offset + nbytes + local_header.size()); //offset of start of global headers, since global header now starts after newly written array
        footer += (unsigned short) 0; //zip file comment length

        //write everything      
        fwrite(&local_header[0],sizeof(char),local_header.size(),fp);
        fwrite(&npy_header[0],sizeof(char),npy_header.size(),fp);
        fwrite(data,sizeof(T),nels,fp);
        fwrite(&global_header[0],sizeof(char),global_header.size(),fp);
        fwrite(&footer[0],sizeof(char),footer.size(),fp);
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
                                      const unsigned int ndims) {

        std::vector<char> dict;
        dict += "{'descr': '";
        dict += BigEndianTest();
        dict += mapType(typeid(T));
        dict += tostring(sizeof(T));
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

        return header;
    }


}

#endif
