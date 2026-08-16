
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <vector>
#include <string>

struct TestStruct {
    int         number;
    float       realNumber;
    std::string str;
};

void WriteJsonInto( TestStruct data, std::stringstream & ss ) {
    ss << "{";
    ss << "number:\"" << std::to_string(data.number)<< "\",";
    ss << "realNumber:\"" << std::to_string(data.realNumber)<< "\",";
    ss << "str:\"" << data.str << "\",";
    ss << "}";
}

// Parse args.
int main() {
    TestStruct data;
    data.number = 2;
    data.realNumber = 5.0f;
    data.str = "GG";

    std::stringstream ss;
    WriteJsonInto(data, ss);

    std::cout << ss.str() << std::endl;
    return 0;
}



