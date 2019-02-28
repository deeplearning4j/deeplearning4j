//
// Created by raver on 2/28/2019.
//

#ifndef DEV_TESTS_PARAMETERS_H
#define DEV_TESTS_PARAMETERS_H

#include <map>
#include <string>
#include <vector>

namespace nd4j {
    class Parameters {
    private:
        std::map<std::string, int> _intParams;
        std::map<std::string, bool> _boolParams;
        std::map<std::string, std::vector<int>> _arrayParams;
    public:
        Parameters() = default;

        Parameters* addIntParam(std::string &string, int param);
        Parameters* addIntParam(std::initializer_list<std::string> strings, std::initializer_list<int> params);

        Parameters* addBoolParam(std::string &string, bool param);
        Parameters* addBoolParam(std::initializer_list<std::string> strings, std::initializer_list<bool> params);

        Parameters* addArrayParam(std::string &string, std::initializer_list<int> param);
        Parameters* addArrayParam(std::initializer_list<std::string> strings, std::initializer_list<std::initializer_list<int>> params);

        int getIntParam(std::string &string);
        bool getBoolParam(std::string &string);
        std::vector<int> getArrayParam(std::string &string);
    };
}

#endif //DEV_TESTS_PARAMETERS_H
