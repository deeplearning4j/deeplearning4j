//
// Created by GS <sgazeos@gmail.com> 3/7/2018
//

#include <graph/GraphUtils.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>

#ifndef _WIN32
#include <sys/types.h>
#include <sys/wait.h>
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

#ifndef _WIN32
    int pipefd[2];
    pipe(pipefd);
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
    nd4j_printf("Run preprocessor as \ncpp %s\n", input);
//    int err;
//    char* cxx_path = getenv("CXX_PATH");
//    if (cxx_path == NULL) {
//        nd4j_printf("Cannot retrieve mandatory environment variable 'CXX_PATH'. Please set up the variable and try again.", "");
//        exit(2);
//    }

    char* cxx = getenv("CXX");
    if (cxx == NULL) {
        nd4j_printf("Cannot retrieve mandatory environment variable 'CXX'. Please set up the variable and try again.", "");
        exit(3);
    }

    std::string pathStr("PATH=/usr/bin:/usr/local/bin:$PATH");
//    pathStr += cxx_path;

    nd4j_printf("%s\n", pathStr.c_str());
    char const* env[] = {// "HOME=/tmp", 
                          pathStr.c_str(),
                          (char *)0 };

// to retrieve c++ version (hardcoded 6): c++ -v 2>&1 | tail -1 | awk '{v = int($3); print v;}' 
    nd4j_printf("Run: \n\t g++ -E -P -std=c++11 -o %s -I{../include/*, ../blas} %s\n", output, input);
    int err = execle(cxx, "g++", "-E", "-P", "-std=c++11", "-o", output, 
        "-I../include",
        "-I../blas",
        "-I../include/ops",
        "-I../include/helpers",
        "-I../include/types",
        "-I../include/array",
        "-I../include/cnpy",
        "-I../include/ops/declarable", 
        input,
        (char*)nullptr, 
        env);

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
