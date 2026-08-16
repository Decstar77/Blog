
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <vector>
#include <string>
#include <concepts>
#include <meta>

struct TestSubStruct {
    bool    boolean;
    float   otherNumber;
};

struct TestStruct {
    int             number;
    float           realNumber;
    std::string     str;
    TestSubStruct   sub;
};

void WriteJsonInto1( TestStruct data, std::stringstream & ss ) {
    ss << "{";
    ss << "number:\"" << std::to_string(data.number)<< "\",";
    ss << "realNumber:\"" << std::to_string(data.realNumber)<< "\",";
    ss << "str:\"" << data.str << "\"";
    ss << "}";
}

template<typename _type_> requires requires ( const _type_ & s ) { std::to_string(s); }
static std::string ToString( const _type_ & s ) {
    return std::to_string(s);
}

static std::string ToString( const std::string & s ) {
    return s;
}

template<typename _type_>
concept HasToString = requires ( const _type_ & s ) {
    { ToString(s) } -> std::convertible_to<std::string>;
};

void WriteJsonInto2( TestStruct data, std::stringstream & ss ) {
    constexpr auto TestType = ^^TestStruct;
    static constexpr auto testMembers = std::define_static_array( std::meta::nonstatic_data_members_of( TestType, std::meta::access_context::current()) );
    ss << "{";

    template for ( constexpr auto m : testMembers ) {
        if constexpr ( HasToString<typename [:std::meta::type_of(m):]> ) {
            ss << "\"" << std::meta::identifier_of(m) << "\":\"" << ToString(data.[:m:]) << "\",";
        }
    };

    ss.seekp(-1, std::ios_base::end);
    ss << "}";
}

template<typename _type_>
void WriteJsonInto3( _type_ data, std::stringstream & ss ) {
    constexpr auto Type = ^^_type_;
    static constexpr auto testMembers = std::define_static_array( std::meta::nonstatic_data_members_of( Type, std::meta::access_context::current()) );
    ss << "{";

    template for ( constexpr auto m : testMembers ) {
        ss << "\"" << std::meta::identifier_of(m) << "\":";
        if constexpr ( HasToString<typename [:std::meta::type_of(m):]> ) {
            ss << "\"" << ToString(data.[:m:]) << "\"";
        } else {
            WriteJsonInto3(data.[:m:], ss);
        }
        ss << ",";
    };

    ss.seekp(-1, std::ios_base::end);
    ss << "}";
}


// Parse args.
int main() {
    TestStruct data;
    data.number = 2;
    data.realNumber = 5.0f;
    data.str = "GG";
    data.sub.boolean = true;
    data.sub.otherNumber = 7.0f;

    std::stringstream ss;
    WriteJsonInto3(data, ss);

    std::cout << ss.str() << std::endl;

    std::ofstream file("test.json");
    if ( file.is_open() ) {
        file << ss.str();
        file.close();
    }

    return 0;
}



