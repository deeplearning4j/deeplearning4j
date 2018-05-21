/*
 * GraphOpt class declarations
 *
 * GraphOpt class used for parsing command line arguments 
 * 
 *
 * Created by GS <sgazeos@gmail.com> 3/2/2018
 *
 */

#ifndef __H__GRAPH_OPTIONS__
#define __H__GRAPH_OPTIONS__

#include <string>
#include <list>
#include <unordered_map>
#include <iostream>
#include <algorithm>

class GraphOpt {
public:
    typedef std::list<std::string> FileList;
    typedef std::list<int> OptionList;
    typedef std::unordered_map<int, std::string> ArgumentDict;
public:
    GraphOpt()
    {}

    static int optionsWithArgs(int argc, char* argv[], GraphOpt& options);

    FileList& files() { return _files; }
    FileList const& files() const { return _files; } 
    OptionList const& options() const { return _opts; } 
    std::string outputName() const { return _args.at('o'); }
    std::string arch() const {
        if (_args.count('a') < 1) {
            printf("No Arg!!!\n");
            fflush(stdout);
        }
        return _args.at('a'); 
    };
    std::ostream& help(std::string app, std::ostream& out);
    bool hasParam(int param) const { return std::find(_opts.begin(), _opts.end(), param) != _opts.end(); }
    
    friend std::ostream& operator<< (std::ostream& out, GraphOpt const& opts);

    void reset() {
        _files.clear();
        _opts.clear();
        _args.clear();
    }

private:
    FileList _files;
    OptionList _opts;
    ArgumentDict _args;
};

#endif
