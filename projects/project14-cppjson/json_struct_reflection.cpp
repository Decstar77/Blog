
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <vector>
#include <string>
#include <string_view>
#include <charconv>
#include <concepts>
#include <ranges>
#include <type_traits>
#include <meta>

struct Vec3 {
    float x;
    float y;
    float z;
};

struct TestSubStruct {
    bool    boolean;
    float   otherNumber;
};

struct TestStruct {
    int                 number;
    float               realNumber;
    std::string         str;
    TestSubStruct       sub;
    std::vector<int>    ids;
    std::vector<Vec3>   vecs;
};

/*
================================================================
================================================================
================================================================
*/
void WriteJsonInto1( TestStruct data, std::stringstream & ss ) {
    ss << "{";
    ss << "number:\"" << std::to_string(data.number)<< "\",";
    ss << "realNumber:\"" << std::to_string(data.realNumber)<< "\",";
    ss << "str:\"" << data.str << "\"";
    ss << "}";
}

/*
================================================================
================================================================
*/
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

/*
================================================================
================================================================
*/
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

/*
================================================================
================================================================
*/
static void WriteJsonString( const std::string & s, std::stringstream & ss ) {
    ss << "\"";
    for ( const char c : s ) {
        switch ( c ) {
            case '\"': ss << "\\\""; break;
            case '\\': ss << "\\\\"; break;
            case '\b': ss << "\\b";  break;
            case '\f': ss << "\\f";  break;
            case '\n': ss << "\\n";  break;
            case '\r': ss << "\\r";  break;
            case '\t': ss << "\\t";  break;
            default: {
                ss << c; 
            } break;
        }
    }
    ss << "\"";
}

template<typename _type_> static void WriteJsonValue( const _type_ & value, std::stringstream & ss );
template<typename _type_> static void WriteJsonObject( const _type_ & value, std::stringstream & ss );

template<typename _type_>
static void WriteJsonValue( const _type_ & value, std::stringstream & ss ) {
    if constexpr ( std::is_same_v<_type_, bool> ) {
        ss << ( value ? "true" : "false" );
    } else if constexpr ( std::is_arithmetic_v<_type_> ) {
        ss << ToString( value );
    } else if constexpr ( HasToString<_type_> ) {
        WriteJsonString( ToString( value ), ss );
    } else if constexpr ( std::ranges::range<_type_> ) {
        ss << "[";
        bool first = true;
        for ( const auto & element : value ) {
            if ( !first ) { ss << ","; }
            first = false;
            WriteJsonValue( element, ss );
        }
        ss << "]";
    } else {
        WriteJsonObject( value, ss );
    }
}

template<typename _type_>
static void WriteJsonObject( const _type_ & value, std::stringstream & ss ) {
    constexpr auto Type = ^^_type_;
    static constexpr auto members = std::define_static_array( std::meta::nonstatic_data_members_of( Type, std::meta::access_context::current()) );

    ss << "{";
    bool first = true;
    template for ( constexpr auto m : members ) {
        if ( first == false ) { ss << ","; }
        first = false;
        ss << "\"" << std::meta::identifier_of(m) << "\":";
        WriteJsonValue( value.[:m:], ss );
    };
    ss << "}";
}

template<typename _type_>
void WriteJsonInto4( const _type_ & data, std::stringstream & ss ) {
    WriteJsonValue( data, ss );
}

// Parse args.
int main() {
    TestStruct data;
    data.number = 2;
    data.realNumber = 5.0f;
    data.str = "GG";
    //data.str = "GG \"quoted\" back\\slash\nnewline\ttab \x01 caf\xc3\xa9";
    data.sub.boolean = true;
    data.sub.otherNumber = 7.0f;
    data.ids = { 1, 2, 3 };
    data.vecs = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } };

    std::stringstream ss;
    WriteJsonInto4(data, ss);

    std::cout << ss.str() << std::endl;

    std::ofstream file("test.json");
    if ( file.is_open() ) {
        file << ss.str();
        file.close();
    }

    return 0;
}



