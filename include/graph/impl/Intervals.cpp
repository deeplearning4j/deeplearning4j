//
// @author yurii@skymind.io
//
#include <graph/Intervals.h>

namespace nd4j {

    // default constructor
    Intervals::Intervals(): _content({{}}) {}
        
    // constructor
    Intervals::Intervals(const std::initializer_list<std::vector<int>>& content ): _content(content) {}
    
    //////////////////////////////////////////////////////////////////////////
    // accessing operator
    std::vector<int> Intervals::operator[](const int i) const {
        
        return *(_content.begin() + i);
    }

    //////////////////////////////////////////////////////////////////////////
    // returns size of _content
    int Intervals::size() const {
    
        return _content.size();
    }

    //////////////////////////////////////////////////////////////////////////
    // modifying operator     
    // std::vector<int>& Intervals::operator()(const int i) {
    //     return _content[i];
    // }


}

