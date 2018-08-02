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
// Created by GS <sgazeos@gmail.com> 3/7/2018
//

#include <graph/GraphUtils.h>
#include <cstdlib>
#include <cstdio>

#ifdef __linux__ //_WIN32
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <linux/limits.h>
//#eldef __APPLE__
//#include <sys/types.h>
//#include <sys/wait.h>
#endif
namespace nd4j {
namespace graph {

bool 
GraphUtils::filterOperations(GraphUtils::OpList& ops) {
    bool modified = false;

    std::vector<OpDescriptor> filtered(ops);

    std::sort(filtered.begin(), filtered.end(), [](OpDescriptor a, OpDescriptor b) {
        return a.getOpName()->compare(*(b.getOpName())) < 0;
    });
    std::string name = *(filtered[0].getOpName());

    for (int e = 1; e < filtered.size(); e++) {
//        nd4j_printf(">%s<, %lu %lu\n", name.c_str(), ops.size(), filtered.size());
        if (0 == filtered[e].getOpName()->compare(name)) {
            // there is a match
            auto fi = std::find_if(ops.begin(), ops.end(), 
                [name](OpDescriptor a) { 
                    return a.getOpName()->compare(name) == 0; 
            });
            if (fi != ops.end())
                ops.erase(fi);
            modified = true;
        }
        name = *(filtered[e].getOpName());
    }
    return modified;
}

std::string 
GraphUtils::makeCommandLine(GraphUtils::OpList& ops) {
    std::string res;

    if (!ops.empty()) {
        res += std::string(" -g \"-DLIBND4J_OPS_LIST='");
        //res += *(ops[0].getOpName());
        for (int i = 0; i < ops.size(); i++) {
            res += std::string("-DOP_");
            res += *(ops[i].getOpName());
            res += "=true ";
        }
        res += "'\"";
    }

    return res;
}

int 
GraphUtils::runPreprocessor(char const* input, char const* output) {
    int status = 0;

#ifdef __linux__ //_WIN32
    int pipefd[2];
    status = pipe(pipefd);
    pid_t pid = fork();
    if (pid == 0)
    {
        close(pipefd[0]);    // close reading end in the child
    
        dup2(pipefd[1], 1);  // send stdout to the pipe
        dup2(pipefd[1], 2);  // send stderr to the pipe

        close(pipefd[1]);    // this descriptor is no longer needed

    #if __CNUC__ < 4 && __GNUC_MINOR__ < 9
    #pragma error "Compiler version should be greater then 4.9"
    #endif

    // just stacking everything together
//    std::string cmdline = "./buildnativeoperations.sh " + 
///        std::string(name_arg) + 
//        std::string(build_arg) + 
///        std::string(arch_arg) + 
//        std::string(opts_arg);

    FILE *f = popen("which c++", "r");
    if(f == NULL) {
        std::cerr << "Cannot find c++ compiler with 'which' command." << std::endl;
        exit(1);
    }
#if _POSIX_C_SOURCE >= 200809L
    char* line = nullptr;
    size_t size = 0;
    ssize_t len;

    if ((len = getdelim(&line, &size, '\n', f)) < 2) {
        std::cerr << "Cannot find c++ compiler with 'which' command." << std::endl;
        exit(2);
    }
    if (line[len - 1] == '\n')
        line[len - 1] = '\0'; 

    std::string cmd(line);

    fclose(f);

    free(line);
#else
    std::string cmd;
    {
        
        char szLine[PATH_MAX];
        if (NULL == fgets(szLine, sizeof(szLine), f)) {
            std::cerr << "Cannot find c++ compiler with 'which' command." << std::endl;
            exit(3);
        }
        char* p = strchr(szLine, '\n');
        if (p) {
            *p = '\0';
        }
        cmd = szLine;
    }
#endif

    char const* cxx = cmd.c_str(); //;getenv("CXX");
//    if (cxx == nullptr) {
//        nd4j_printf("Cannot retrieve mandatory environment variable 'CXX'. Please set up the variable and try again.", "");
//        exit(3);
//    }
    //char* pathEnv = getenv("PATH");
    //std::string pathStr("PATH=./;");
    //pathStr += pathEnv;

    //nd4j_printf("%s\n", pathStr.c_str());
//    char const* env[] = {// "HOME=/tmp", 
//                          pathStr.c_str(),
//                          (char *)0 };

// to retrieve c++ version (hardcoded 6): c++ -v 2>&1 | tail -1 | awk '{v = int($3); print v;}' 

    std::vector<char*> params;//(9);
    std::vector<std::string> args;//(9);
    args.emplace_back(cmd);
    args.emplace_back(std::string("-E"));
    args.emplace_back(std::string("-P"));
    args.emplace_back(std::string("-std=c++11"));
    args.emplace_back(std::string("-o"));
    args.emplace_back(output);
    args.emplace_back(std::string("-I../include"));
    args.emplace_back(std::string("-I../blas"));
    args.emplace_back(std::string("-I../include/ops"));
    args.emplace_back(std::string("-I../include/helpers"));
    args.emplace_back(std::string("-I../include/types"));
    args.emplace_back(std::string("-I../include/array"));
    args.emplace_back(std::string("-I../include/cnpy"));
    args.emplace_back(std::string("-I../include/ops/declarable")); 
    args.emplace_back(input);

    std::string preprocessorCmd(cxx);
    bool skip = true;
    for (auto& arg: args) {
        if (!skip) {
            preprocessorCmd += ' ';
            preprocessorCmd += arg;
        }
        else 
            skip = false;
        params.emplace_back(const_cast<char*>(arg.data()));
    }
    params.emplace_back(nullptr);
    nd4j_printf("Run: \n\t %s\n", preprocessorCmd.c_str());

    int err = execvp(cmd.c_str(), &params[0]);

    if (err < 0) {
        perror("\nCannot run Preprocessor properly due \n");
    }
    status = err;
    nd4j_printf("Header file %s was generated.\n", output);
//    nd4j_printf("Running build script\n%s\n", cmdline.c_str());
    }
    else
    {
    // parent
        char buffer[1024];
        close(pipefd[1]);  // close the write end of the pipe in the parent
        memset(buffer, 0, sizeof(buffer));
        while (read(pipefd[0], buffer, sizeof(buffer)) != 0)  {
            printf("%s\n", buffer);
        }
        waitpid(pid, &status, 0);
    }
#endif
    return status;
}

}
}
