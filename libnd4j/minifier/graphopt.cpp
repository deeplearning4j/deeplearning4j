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
 * Implementation for GraphOpt class.
 *
 * Created by GS <sgazeos@gmail.com> 3/2/2018.
 *
 */

#include <cstdlib>
#include <cstring>

#include "graphopt.h"

std::ostream& 
operator<< (std::ostream& out, GraphOpt const& opts) {
    if (opts._files.empty() && opts._opts.empty()) {
        out << "Empty options" << std::endl;
        return out;
    }
    out << "==================================================" << std::endl;
    out << "Files:" << std::endl;
    int index = 1;
    for (auto file: opts._files) {
        out << "File " << index++ << ": " << file << std::endl;
    }
    out << "Options:" << std::endl;
    for (char opt: opts._opts) {
        out << "Option: " << opt;
        if (opts._args.find(opt) != opts._args.end()) {
            out << " with arg: " << opts._args.at(opt) << std::endl;
        }
        else {
            out << std::endl;
        }
    }
    out << "==================================================";
    return out;
}

////////////////////////////////////////////////////////////////////////////////
int 
GraphOpt::optionsWithArgs(int argc, char* argv[], GraphOpt& res) {
    char* optArg = nullptr;
    int optIndex = 1;
    
    char const* optionStr = "lxa:o:e";
    std::string const defaultOutputName("nd4jlib_mini");

    for (optIndex = 1; (optIndex < argc) && (argv[optIndex][0] == '-') && 
                       (argv[optIndex][0]); optIndex++) {

        int opt = argv[optIndex][1];

        if (opt == '?' || opt == 'h') {
            res.help(argv[0], std::cout);
            res.reset();
            return 1;
        }

        char const* p = strchr(optionStr, opt);

        if (p == nullptr)
        {
            std::cerr << "opt " << (char)opt << " not found with " << optionStr << std::endl;
            res._opts.push_back('?');
            res.reset();
            return -1;
        }
        else {
            res._opts.push_back(opt);

            if (p[1] == ':') // processing param with 
            {
                optIndex++;
                if (optIndex >= argc)
                {
                    std::cerr << "optIndex " << optIndex << " is out of bounds " << argc << std::endl;
                    res.reset();
                    res._opts.push_back('?');
                    return -2;
                }
                res._args[opt] = std::string(argv[optIndex]);
            }
        }
    }

    if ( !res.hasParam('l') && !res.hasParam('x') ) {
        std::cerr << "No -l or -x params are provided. At least one of them should be used." << std::endl;
        res.reset();
        res._opts.push_back('?');
        return -3;
    }

    if (res._args.empty())
        res._args['o'] = defaultOutputName;

    for ( ; optIndex < argc; optIndex++) {
        res._files.push_back(std::string(argv[optIndex]));
    }
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
std::ostream& 
GraphOpt::help(std::string app, std::ostream& out) {
    out << "Usage: \n" << app << " [-lxe] [-o outname] filename1 "
                            "[filename2 filename3 ... filenameN]" << std::endl;
    out << "Parameters:" << std::endl;
    out << "\t-l\t Generate library" << std::endl;
    out << "\t-x\t Generate executable" << std::endl;
    out << "\t-e\t Embed the Graph(s) into executable as resource" << std::endl;
    out << "\t-o <name> Set up output name (for library, executable or both)" << std::endl;
    out << "\t-a <arch> target CPU architecture" << std::endl; 
    out << "\t-h\t This help" << std::endl;

    return out;
}

