#include <sys/stat.h>

#ifdef _WIN32
    #include <io.h>
#else
    #include <unistd.h>
#endif
#include <cstdlib>
#include "graphopt.h"
#include <GraphExecutioner.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/GraphUtils.h>

int
main(int argc, char *argv[]) {
    // this string will contain list of operations
    std::string opts_arg;

    // this string will contain optional name for output binary file
    std::string name_arg;

    // this string will contain binary compilation mode: shared/static/executable
    std::string build_arg;

    // this string will contain target arch/optimization mode
    std::string arch_arg;

    GraphOpt opt;
    int err = GraphOpt::optionsWithArgs(argc, argv, opt);
    
    //std::cout << opt << std::endl;
    if (err > 0) {   
        // only help message
        return err;
    }

    if (err < 0) {
        std::cerr << "Wrong parameter list" << std::endl;
        opt.help(argv[0], std::cerr); 
        return err;
    }
    
    for (int option: opt.options()) {
        std::cout << "Option \'" << (char)option <<"\': ";
        switch (option) {
        case 'l':
            std::cout << "Build library" << std::endl;
            break;
        case 'x':
            std::cout << "Build executable" << std::endl;
            break;
        case 'e':
            std::cout << "Link the Graph to executable as Resource" << std::endl;
            break;
        case 'o':
            std::cout << "Output file name is " << opt.outputName() << std::endl;
            break;
        case 'a':
            std::cout << "Target arch: " << opt.arch() << std::endl;
            break;
        default:
            std::cerr << "Wrong parameter " << (char)option << std::endl;
        }
    }
    
    if (!opt.hasParam('o')) {
        std::cout << "Ouput file name is " << opt.outputName() << std::endl;
    }

    name_arg = " --name \'" + opt.outputName() + "\' ";

    if (opt.hasParam('a'))
        arch_arg = opt.arch();

    std::vector<OpDescriptor> descriptors;
    nd4j_printf("Total available operations: %i\n", OpRegistrator::getInstance()->numberOfOperations());

    for (auto file: opt.files()) {
        // all files will be checked for accessibility & size
#ifdef _WIN32
        if (_access(file.c_str(), 1) != -1) {
#else
        if (access(file.c_str(), F_OK | R_OK) != -1) {
#endif
#ifdef _WIN32
            struct _stat st;
            _stat(file.c_str(), &st);
#else
            struct stat st;
            stat(file.c_str(), &st);
#endif  
            if (st.st_size != 0) {
                //std::cout << "File " << file << " exists and can be read" << std::endl;
                auto graph = GraphExecutioner<float>::importFromFlatBuffers(file.c_str());
                auto ops = graph->getOperations();

                for (auto &v:ops) {
                    descriptors.emplace_back(v);
                }
            } else {
                std::cerr << "File " << file << " exists, but has zero size" << std::endl;
                return 2;
            }
        }
        else {
            std::cerr << "File " << file << " does not exists " << std::endl;
            return 10;
        }
    }

    if (!descriptors.empty()) {
        GraphUtils::filterOperations(descriptors);

        nd4j_printf("Operations found so far:\n","");
        for (auto &v: descriptors) {
            nd4j_printf("%s\n", v.getOpName()->c_str());
        }

        // building list of operations
        opts_arg = GraphUtils::makeCommandLine(descriptors);
    }
    nd4j_printf("\n","");

#ifdef __GNUC__
    #if __CNUC__ < 4 && __GNUC__ > 6
    #pragma error "Compiler version should be between 4.9 and 6.4"
    #endif
    // just stacking everything together
    std::string cmdline = "./buildnativeoperations.sh " + name_arg + build_arg + arch_arg + opts_arg;
    std::string input("../include/ops/declarable/CustomOperations.h");
    std::string output(opt.outputName());
    nd4j_printf("Run preprocessor as \ncpp %s\n", input.c_str());
//    int err;
    char* cxx = getenv("CXX");
    if (cxx == NULL) {
        nd4j_printf("Cannot retrieve mandatory environment variable %s. Please set up the variable and try again.", "CXX");
        return 1;
    }
    std::string pathStr("PATH=/usr/bin:/usr/local/bin:");
    pathStr += cxx;
    std::string gccVersion = std::to_string(__GNUC__);
//    pathStr += gccVersion;
    std::string includeStr("CPLUS_INCLUDE_PATH=/usr/include/c++/");
    includeStr += gccVersion; // GCC version is here
//    includeStr += ":/usr/include/x86_64-linux-gnu/c++/";
    includeStr += gccVersion;

    nd4j_printf("%s\n", pathStr.c_str());
    char const* env[] = { "HOME=/tmp", 
                          "LOGNAME=minifier", 
                          //"PATH=/usr/bin:/usr/local/bin:/usr/lib/gcc/x86_64-linux-gnu/6", 
                          pathStr.c_str(),
                          includeStr.c_str(), //"CPLUS_INCLUDE_PATH=/usr/include/c++/6:/usr/include/x86_64-linux-gnu/c++/6",
                          (char *)0 };

// to retrieve c++ version (hardcoded 6): c++ -v 2>&1 | tail -1 | awk '{v = int($3); print v;}' 
    err = execle(std::string("/usr/bin/cpp-" + gccVersion).c_str(), std::string("cpp-" + gccVersion).c_str(), "-x", "c++", "-std=c++11", "-P", "-o", output.c_str(), 
        "-I../include",
        "-I../blas",
        "-I../include/ops",
        "-I../include/helpers",
        "-I../include/types",
        "-I../include/array",
        "-I../include/cnpy",
        "-I../include/ops/declarable", 
        input.c_str(),
        (char*)nullptr, 
        env);
    if (err < 0) {
        perror("\nCannot run CPP properly due \n");
    }
#endif

    //nd4j_printf("Command line: %s\n", cmdline.c_str());
    // FIXME: do this in cross-platform way
    nd4j_printf("Building minified library...\n", "");
    //system(cmdline.c_str());

    return EXIT_SUCCESS;
}
