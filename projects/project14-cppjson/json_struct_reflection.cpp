
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
                if ( static_cast<unsigned char>( c ) < 0x20 ) {
                    // The remaining control characters have no short escape, so spell them out.
                    constexpr char hex[] = "0123456789abcdef";
                    ss << "\\u00" << hex[ ( c >> 4 ) & 0xF ] << hex[ c & 0xF ];
                } else {
                    ss << c; // Everything else, UTF-8 bytes included, passes through unchanged.
                }
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
        ss << ToString( value );                            // Numbers are bare, not quoted.
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

static void AppendUtf8( uint32_t codepoint, std::string & out ) {
    if ( codepoint < 0x80 ) {
        out += static_cast<char>( codepoint );
    } else if ( codepoint < 0x800 ) {
        out += static_cast<char>( 0xC0 | ( codepoint >> 6 ) );
        out += static_cast<char>( 0x80 | ( codepoint & 0x3F ) );
    } else if ( codepoint < 0x10000 ) {
        out += static_cast<char>( 0xE0 | ( codepoint >> 12 ) );
        out += static_cast<char>( 0x80 | ( ( codepoint >> 6 ) & 0x3F ) );
        out += static_cast<char>( 0x80 | ( codepoint & 0x3F ) );
    } else {
        out += static_cast<char>( 0xF0 | ( codepoint >> 18 ) );
        out += static_cast<char>( 0x80 | ( ( codepoint >> 12 ) & 0x3F ) );
        out += static_cast<char>( 0x80 | ( ( codepoint >> 6 ) & 0x3F ) );
        out += static_cast<char>( 0x80 | ( codepoint & 0x3F ) );
    }
}

// A cursor over the document. Every operation returns false and leaves 'error' set on the first
// problem, so callers can bail without checking a separate flag.
struct JsonReader {
    std::string_view    text;
    size_t              pos     = 0;
    std::string         error   = "";

    bool Fail( const std::string & message ) {
        if ( error.empty() ) { error = message + " at offset " + std::to_string( pos ); }
        return false;
    }

    void SkipWhitespace() {
        while ( pos < text.size() ) {
            const char c = text[pos];
            if ( c != ' ' && c != '\t' && c != '\n' && c != '\r' ) { break; }
            pos++;
        }
    }

    char Peek() {
        SkipWhitespace();
        return ( pos < text.size() ) ? text[pos] : '\0';
    }

    bool AtEnd() {
        SkipWhitespace();
        return pos >= text.size();
    }

    bool Expect( char c ) {
        if ( Peek() != c ) { return Fail( std::string( "expected '" ) + c + "'" ); }
        pos++;
        return true;
    }

    bool Literal( std::string_view word ) {
        SkipWhitespace();
        if ( text.substr( pos ).starts_with( word ) == false ) {
            return Fail( "expected '" + std::string( word ) + "'" );
        }
        pos += word.size();
        return true;
    }

    bool ReadHex4( uint32_t & out ) {
        if ( pos + 4 > text.size() ) { return Fail( "truncated \\u escape" ); }
        out = 0;
        for ( int i = 0; i < 4; i++ ) {
            const char c = text[pos++];
            uint32_t digit = 0;
            if      ( c >= '0' && c <= '9' ) { digit = static_cast<uint32_t>( c - '0' ); }
            else if ( c >= 'a' && c <= 'f' ) { digit = static_cast<uint32_t>( c - 'a' + 10 ); }
            else if ( c >= 'A' && c <= 'F' ) { digit = static_cast<uint32_t>( c - 'A' + 10 ); }
            else { return Fail( "bad hex digit in \\u escape" ); }
            out = ( out << 4 ) | digit;
        }
        return true;
    }

    bool ReadString( std::string & out ) {
        if ( Expect( '"' ) == false ) { return false; }
        out.clear();
        while ( true ) {
            if ( pos >= text.size() ) { return Fail( "unterminated string" ); }
            const char c = text[pos++];
            if ( c == '"' )  { return true; }
            if ( c != '\\' ) { out += c; continue; }
            if ( pos >= text.size() ) { return Fail( "unterminated escape" ); }

            const char escape = text[pos++];
            switch ( escape ) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    uint32_t codepoint = 0;
                    if ( ReadHex4( codepoint ) == false ) { return false; }
                    if ( codepoint >= 0xD800 && codepoint <= 0xDBFF ) {
                        // High surrogate, so the low half has to follow as its own escape.
                        if ( text.substr( pos ).starts_with( "\\u" ) == false ) { return Fail( "expected low surrogate" ); }
                        pos += 2;
                        uint32_t low = 0;
                        if ( ReadHex4( low ) == false ) { return false; }
                        if ( low < 0xDC00 || low > 0xDFFF ) { return Fail( "invalid low surrogate" ); }
                        codepoint = 0x10000 + ( ( codepoint - 0xD800 ) << 10 ) + ( low - 0xDC00 );
                    } else if ( codepoint >= 0xDC00 && codepoint <= 0xDFFF ) {
                        return Fail( "unpaired low surrogate" );
                    }
                    AppendUtf8( codepoint, out );
                } break;
                default: return Fail( "unknown escape" );
            }
        }
    }

    template<typename _type_>
    bool ReadNumber( _type_ & out ) {
        SkipWhitespace();
        const char * first  = text.data() + pos;
        const char * last   = text.data() + text.size();
        const auto   result = std::from_chars( first, last, out );
        if ( result.ec != std::errc() ) { return Fail( "invalid number" ); }
        pos = static_cast<size_t>( result.ptr - text.data() );
        return true;
    }

    bool SkipNumber() {
        SkipWhitespace();
        const size_t start = pos;
        while ( pos < text.size() ) {
            const char c = text[pos];
            const bool isNumberChar = ( c >= '0' && c <= '9' ) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E';
            if ( isNumberChar == false ) { break; }
            pos++;
        }
        return ( pos > start ) ? true : Fail( "expected a value" );
    }

    // Used for keys the target struct doesn't have. Consumes exactly one value, however nested.
    bool SkipValue() {
        const char c = Peek();
        if ( c == '"' ) { std::string ignored; return ReadString( ignored ); }
        if ( c == 't' ) { return Literal( "true" ); }
        if ( c == 'f' ) { return Literal( "false" ); }
        if ( c == 'n' ) { return Literal( "null" ); }
        if ( c != '{' && c != '[' ) { return SkipNumber(); }

        const bool isObject = ( c == '{' );
        const char close    = isObject ? '}' : ']';
        pos++;
        if ( Peek() == close ) { pos++; return true; }
        while ( true ) {
            if ( isObject ) {
                std::string key;
                if ( ReadString( key ) == false ) { return false; }
                if ( Expect( ':' ) == false )     { return false; }
            }
            if ( SkipValue() == false ) { return false; }
            if ( Peek() == ',' ) { pos++; continue; }
            return Expect( close );
        }
    }
};

// The mirror of ToString. Overload it for any type that serialises as a JSON string.
static bool FromString( const std::string & s, std::string & out ) {
    out = s;
    return true;
}

template<typename _type_>
concept HasFromString = requires ( const std::string & s, _type_ & out ) {
    { FromString( s, out ) } -> std::same_as<bool>;
};

template<typename _type_> static bool ReadJsonValue( JsonReader & reader, _type_ & out );
template<typename _type_> static bool ReadJsonObject( JsonReader & reader, _type_ & out );

template<typename _type_>
static bool ReadJsonValue( JsonReader & reader, _type_ & out ) {
    if ( reader.Peek() == 'n' ) { return reader.Literal( "null" ); }   // null leaves 'out' as it was.

    if constexpr ( std::is_same_v<_type_, bool> ) {
        const char c = reader.Peek();
        if ( c == 't' ) { out = true;  return reader.Literal( "true" ); }
        if ( c == 'f' ) { out = false; return reader.Literal( "false" ); }
        return reader.Fail( "expected a boolean" );
    } else if constexpr ( std::is_arithmetic_v<_type_> ) {
        return reader.ReadNumber( out );
    } else if constexpr ( HasFromString<_type_> ) {
        std::string text;
        if ( reader.ReadString( text ) == false )   { return false; }
        if ( FromString( text, out ) == false )     { return reader.Fail( "could not convert string" ); }
        return true;
    } else if constexpr ( std::ranges::range<_type_> ) {
        if ( reader.Expect( '[' ) == false ) { return false; }
        out.clear();
        if ( reader.Peek() == ']' ) { reader.pos++; return true; }
        while ( true ) {
            std::ranges::range_value_t<_type_> element = {};
            if ( ReadJsonValue( reader, element ) == false ) { return false; }
            out.push_back( std::move( element ) );
            if ( reader.Peek() == ',' ) { reader.pos++; continue; }
            return reader.Expect( ']' );
        }
    } else {
        return ReadJsonObject( reader, out );
    }
}

template<typename _type_>
static bool ReadJsonObject( JsonReader & reader, _type_ & out ) {
    constexpr auto Type = ^^_type_;
    static constexpr auto members = std::define_static_array( std::meta::nonstatic_data_members_of( Type, std::meta::access_context::current()) );

    if ( reader.Expect( '{' ) == false ) { return false; }
    if ( reader.Peek() == '}' ) { reader.pos++; return true; }

    while ( true ) {
        std::string key;
        if ( reader.ReadString( key ) == false ) { return false; }
        if ( reader.Expect( ':' ) == false )     { return false; }

        // The write side turns a member into a name; here we walk the same list and match back.
        bool matched = false;
        template for ( constexpr auto m : members ) {
            if ( matched == false && key == std::meta::identifier_of(m) ) {
                matched = true;
                if ( ReadJsonValue( reader, out.[:m:] ) == false ) { return false; }
            }
        };
        if ( matched == false && reader.SkipValue() == false ) { return false; } // Unknown keys are ignored.

        if ( reader.Peek() == ',' ) { reader.pos++; continue; }
        return reader.Expect( '}' );
    }
}

template<typename _type_>
bool ReadJson( std::string_view text, _type_ & out, std::string * error = nullptr ) {
    JsonReader reader = { text };
    const bool parsed = ReadJsonValue( reader, out );
    if ( parsed && reader.AtEnd() == false ) { reader.Fail( "trailing characters" ); }
    if ( reader.error.empty() == false ) {
        if ( error != nullptr ) { *error = reader.error; }
        return false;
    }
    return true;
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

    // Round trip: parse what we just wrote, write it back out, and the two should be identical.
    TestStruct  parsed = {};
    std::string error  = "";
    if ( ReadJson( ss.str(), parsed, &error ) == false ) {
        std::cout << "parse failed: " << error << std::endl;
        return 1;
    }

    std::stringstream again;
    WriteJsonInto4(parsed, again);
    std::cout << again.str() << std::endl;
    std::cout << ( ( again.str() == ss.str() ) ? "round trip OK" : "round trip MISMATCH" ) << std::endl;

    return 0;
}



