//
// Created by GS <sgazeos@gmail.com> 3/7/2018
//

#ifndef __H__GRAPH_UTILS__
#define __H__GRAPH_UTILS__

#include <vector>
#include <ops/declarable/OpDescriptor.h>
#include <ops/declarable/DeclarableOp.h>

namespace nd4j {
namespace graph {

class GraphUtils {
public:
    typedef std::vector<OpDescriptor> OpList;

public:
    static bool filterOperations(OpList& ops);
    static std::string makeCommandLine(OpList& ops);
    static int runPreprocessor(char const* name_arg, char const* build_arg, char const* arch_arg, char const* opts_arg, char const* output);
};

}
}
#endif
