//
// @author yurii@skymind.io
//
#include <graph/Intervals.h>

namespace nd4j {

    // default constructor
    Intervals::Intervals(): _content({{}}) {}
        
    // constructor
    Intervals::Intervals(const std::initializer_list<std::vector<Nd4jLong>>& content ): _content(content) {}
    Intervals::Intervals(const std::vector<std::vector<Nd4jLong>>& content ): _content(content) {}
    
    //////////////////////////////////////////////////////////////////////////
    // accessing operator
    std::vector<Nd4jLong> Intervals::operator[](const Nd4jLong i) const {
        
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

