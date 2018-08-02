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

#include <pointercast.h>
#include"cnpy.h"



/**
 *
 * @return
 */
char cnpy::BigEndianTest() {
    unsigned char x[] = {1,0};
    short y = *(short*) x;
    return y == 1 ? '<' : '>';
}

/**
 *
 * @param t
 * @return
 */
char cnpy::mapType(const std::type_info &t) {
    if(t == typeid(float) ) return 'f';
    if(t == typeid(double) ) return 'f';
    if(t == typeid(long double) ) return 'f';

    if(t == typeid(int) ) return 'i';
    if(t == typeid(char) ) return 'i';
    if(t == typeid(short) ) return 'i';
    if(t == typeid(long) ) return 'i';
    if(t == typeid(long long) ) return 'i';

    if(t == typeid(unsigned char) ) return 'u';
    if(t == typeid(unsigned short) ) return 'u';
    if(t == typeid(unsigned long) ) return 'u';
    if(t == typeid(unsigned long long) ) return 'u';
    if(t == typeid(unsigned int) ) return 'u';

    if(t == typeid(bool) ) return 'b';

    if(t == typeid(std::complex<float>) ) return 'c';
    if(t == typeid(std::complex<double>) ) return 'c';
    if(t == typeid(std::complex<long double>) ) return 'c';

    else return '?';
}

/**
 *
 * @param lhs
 * @param rhs
 * @return
 */
template<>
std::vector<char>& cnpy::operator+=(std::vector<char>& lhs, const std::string rhs) {
    lhs.insert(lhs.end(),rhs.begin(),rhs.end());
    return lhs;
}

/**
 *
 * @param lhs
 * @param rhs
 * @return
 */
template<>
std::vector<char>& cnpy::operator+=(std::vector<char>& lhs, const char* rhs) {
    //write in little endian
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for(size_t byte = 0; byte < len; byte++) {
        lhs.push_back(rhs[byte]);
    }
    return lhs;
}

/**
 * Load the whole file in to memory
 * @param path
 * @return
 */
char* cnpy::loadFile(char const *path) {
    char* buffer = 0;
    long length;
    FILE * f = fopen (path, "rb"); //was "rb"

    if (f) {
        fseek (f, 0, SEEK_END);
        length = ftell (f);
        fseek (f, 0, SEEK_SET);
        buffer = (char*) malloc ((length+ 1) * sizeof(char));

        // just getting rid of compiler warning
        Nd4jLong fps = 0;

        if (buffer) {
            fps += fread (buffer, sizeof(char), length, f);
        }

        fclose (f);
    }

    buffer[length] = '\0';
    return buffer;
}


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
void cnpy::parseNpyHeaderStr(std::string header,
                             unsigned int &wordSize,
                             unsigned int *&shape,
                             unsigned int &ndims,
                             bool &fortranOrder) {


    int loc1, loc2;

    //fortran order
    loc1 = header.find("fortranOrder") + 16;
    fortranOrder = (header.substr(loc1,5) == "True" ? true : false);

    //shape
    loc1 = header.find("(");
    loc2 = header.find(")");
    std::string str_shape = header.substr(loc1 + 1,loc2 - loc1 - 1);
    if(str_shape[str_shape.size() - 1] == ',') ndims = 1;
    else ndims = std::count(str_shape.begin(),str_shape.end(),',')+1;
    shape = new unsigned int[ndims];
    for(unsigned int i = 0; i < ndims; i++) {
        loc1 = str_shape.find(",");
        shape[i] = atoi(str_shape.substr(0,loc1).c_str());
        str_shape = str_shape.substr(loc1 + 1);
    }

    //endian, word size, data type
    //byte order code | stands for not applicable.
    //not sure when this applies except for byte array
    loc1 = header.find("descr") + 9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    //char type = header[loc1+1];
    //assert(type == map_type(T));

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    wordSize = atoi(str_ws.substr(0,loc2).c_str());
}

/**
 *
 *
 *
 *
 * @param fp the file to open
 * @param wordSize the size of each element in the array
 * @param shape the pointer to where the shape is stored
 * @param ndims the number of dimensions for the array
 * @param fortranOrder
 */
void cnpy::parseNpyHeader(FILE *fp,
                          unsigned int &wordSize,
                          unsigned int *&shape,
                          unsigned int &ndims,
                          bool &fortranOrder) {
    char buffer[256];
    size_t res = fread(buffer,sizeof(char),11,fp);
    if(res != 11)
        throw std::runtime_error("parse_npy_header: failed fread");
    std::string header = fgets(buffer,256,fp);
    assert(header[header.size() - 1] == '\n');
    cnpy::parseNpyHeaderStr(header,
                            wordSize,
                            shape,
                            ndims,
                            fortranOrder);
}



/**
 *
 * @param fp
 * @param nrecs
 * @param global_header_size
 * @param global_header_offset
 */
void cnpy::parseZipFooter(FILE* fp,
                          unsigned short& nrecs,
                          unsigned int& global_header_size,
                          unsigned int& global_header_offset) {

    std::vector<char> footer(22);
    fseek(fp, -22, SEEK_END);
    size_t res = fread(&footer[0],sizeof(char),22,fp);
    if(res != 22)
        throw std::runtime_error("parse_zip_footer: failed fread");

    unsigned short disk_no, disk_start, nrecs_on_disk, comment_len;
    disk_no = *(unsigned short*) &footer[4];
    disk_start = *(unsigned short*) &footer[6];
    nrecs_on_disk = *(unsigned short*) &footer[8];
    nrecs = *(unsigned short*) &footer[10];
    global_header_size = *(unsigned int*) &footer[12];
    global_header_offset = *(unsigned int*) &footer[16];
    comment_len = *(unsigned short*) &footer[20];

    assert(disk_no == 0);
    assert(disk_start == 0);
    assert(nrecs_on_disk == nrecs);
    assert(comment_len == 0);
}

/**
 * Load the numpy array from the given file.
 * @param fp the file to load
 * @return the loaded array
 */
cnpy::NpyArray cnpy::loadNpyFromFile(FILE *fp) {
    unsigned int *shape;
    unsigned int ndims, wordSize;
    bool fortranOrder;
    cnpy::parseNpyHeader(fp,wordSize,shape,ndims,fortranOrder);
    unsigned long long size = 1; //long long so no overflow when multiplying by word_size
    for(unsigned int i = 0;i < ndims;i++) size *= shape[i];

    cnpy::NpyArray arr;
    arr.wordSize = wordSize;
    arr.shape = std::vector<unsigned int>(shape,shape + ndims);
    arr.data = new char[size * wordSize];
    arr.fortranOrder = fortranOrder;
    size_t nread = fread(arr.data,wordSize,size,fp);
    if(nread != size)
        throw std::runtime_error("load_the_npy_file: failed fread");
    return arr;
}

/**
    *
    * @param data
    * @return
    */
cnpy::NpyArray cnpy::loadNpyFromPointer(char *data)  {
    //move the pointer forward by 11 immitating
    //the seek in loading directly from a file
    data += 11;
    unsigned int *shape;
    unsigned int ndims, wordSize;
    bool fortranOrder;
    cnpy::parseNpyHeaderStr(std::string(data),
                            wordSize,
                            shape,
                            ndims,
                            fortranOrder);
    //the "real" data starts after the \n
    char currChar = data[0];
    while(currChar != '\n') {
        data++;
        currChar = data[0];
    }

    //move pass the \n
    data++;

    unsigned long long size = 1; //long long so no overflow when multiplying by word_size
    for(unsigned int i = 0; i < ndims; i++) size *= shape[i];
    char *cursor = data;
    cnpy::NpyArray arr;
    arr.wordSize = wordSize;
    arr.shape = std::vector<unsigned int>(shape,shape + ndims);
    delete[] shape;
    arr.data = cursor;
    arr.fortranOrder = fortranOrder;
    return arr;
}


/**
 * Load the numpy z archive
 * @param fname the fully qualified path
 * @return the
 */
cnpy::npz_t cnpy::npzLoad(std::string fname) {
    FILE* fp = fopen(fname.c_str(),"rb");

    if(!fp) printf("npz_load: Error! Unable to open file %s!\n",fname.c_str());
    assert(fp);

    cnpy::npz_t arrays;

    while(1) {
        std::vector<char> local_header(30);
        size_t headerres = fread(&local_header[0],sizeof(char),30,fp);
        if(headerres != 30)
            throw std::runtime_error("npz_load: failed fread");

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        unsigned short name_len = *(unsigned short*) &local_header[26];
        std::string varname(name_len,' ');
        size_t vname_res = fread(&varname[0],sizeof(char),name_len,fp);
        if(vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");

        //erase the lagging .npy        
        varname.erase(varname.end() - 4,varname.end());

        //read in the extra field
        unsigned short extra_field_len = *(unsigned short*) &local_header[28];
        if(extra_field_len > 0) {
            std::vector<char> buff(extra_field_len);
            size_t efield_res = fread(&buff[0],sizeof(char),extra_field_len,fp);
            if(efield_res != extra_field_len)
                throw std::runtime_error("npz_load: failed fread");
        }

        // arrays[varname] = loadArrayFromFile(fp);
    }

    fclose(fp);
    return arrays;
}

/**
 * Loads a npz (multiple numpy arrays) file
 * @param fname the file name
 * @param varname
 * @return
 */
cnpy::NpyArray cnpy::npzLoad(std::string fname, std::string varname) {
    FILE *fp = fopen(fname.c_str(),"rb");

    if(!fp) {
        printf("npz_load: Error! Unable to open file %s!\n",fname.c_str());
        abort();
    }

    while(1) {
        std::vector<char> local_header(30);
        size_t header_res = fread(&local_header[0],sizeof(char),30,fp);
        if(header_res != 30)
            throw std::runtime_error("npz_load: failed fread");

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        unsigned short name_len = *(unsigned short*) &local_header[26];
        std::string vname(name_len,' ');
        size_t vname_res = fread(&vname[0],sizeof(char),name_len,fp);
        if(vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");
        vname.erase(vname.end()-4,vname.end()); //erase the lagging .npy

        //read in the extra field
        unsigned short extra_field_len = *(unsigned short*) &local_header[28];
        fseek(fp,extra_field_len,SEEK_CUR); //skip past the extra field

        if(vname == varname) {
            NpyArray array = cnpy::loadNpyFromFile(fp);
            fclose(fp);
            return array;
        }
        else {
            //skip past the data
            unsigned int size = *(unsigned int*) &local_header[22];
            fseek(fp,size,SEEK_CUR);
        }
    }

    fclose(fp);
    printf("npz_load: Error! Variable name %s not found in %s!\n",varname.c_str(),fname.c_str());
    abort();
}




/**
 * Load a numpy array from the given file
 * @param fname the fully qualified path for the file
 * @return the NpArray for this file
 */
cnpy::NpyArray cnpy::npyLoad(std::string fname) {
    FILE* fp = fopen(fname.c_str(), "rb");

    if(!fp) {
        printf("npy_load: Error! Unable to open file %s!\n",fname.c_str());
        abort();
    }

    NpyArray arr = cnpy::loadNpyFromFile(fp);

    fclose(fp);
    return arr;
}



