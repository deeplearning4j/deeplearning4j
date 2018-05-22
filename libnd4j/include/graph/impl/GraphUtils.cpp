//
// Created by GS <sgazeos@gmail.com> 3/7/2018
//

#include <graph/GraphUtils.h>

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

}
}
