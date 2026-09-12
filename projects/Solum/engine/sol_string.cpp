#include "sol_string.h"

#include <cstdlib>
#include <cstring>

namespace sol {

    static bool IsSpace( char c ) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
    }

    static char ToLower( char c ) {
        return ( c >= 'A' && c <= 'Z' ) ? (char)( c - 'A' + 'a' ) : c;
    }

    StringView::StringView( const char * cstr ) {
        data = cstr;
        count = StringLength( cstr );
    }

    StringView::StringView( const char * chars, i32 charCount ) {
        data = chars;
        count = charCount > 0 ? charCount : 0;
    }

    i32 StringLength( const char * cstr ) {
        if( cstr == nullptr ) {
            return 0;
        }

        i32 length = 0;
        while( cstr[length] != '\0' ) {
            length++;
        }
        return length;
    }

    bool StringIsEmpty( StringView str ) {
        return str.count == 0;
    }

    bool StringEquals( StringView a, StringView b ) {
        if( a.count != b.count ) {
            return false;
        }
        if( a.count == 0 ) {
            return true;
        }
        return memcmp( a.data, b.data, (size_t)a.count ) == 0;
    }

    bool StringEqualsIgnoreCase( StringView a, StringView b ) {
        if( a.count != b.count ) {
            return false;
        }

        for( i32 i = 0; i < a.count; i++ ) {
            if( ToLower( a.data[i] ) != ToLower( b.data[i] ) ) {
                return false;
            }
        }
        return true;
    }

    i32 StringCompare( StringView a, StringView b ) {
        const i32 shared = a.count < b.count ? a.count : b.count;
        if( shared > 0 ) {
            const int result = memcmp( a.data, b.data, (size_t)shared );
            if( result != 0 ) {
                return result < 0 ? -1 : 1;
            }
        }

        // Shared prefix is equal, so the shorter one sorts first.
        if( a.count == b.count ) {
            return 0;
        }
        return a.count < b.count ? -1 : 1;
    }

    bool operator==( StringView a, StringView b ) {
        return StringEquals( a, b );
    }

    bool operator!=( StringView a, StringView b ) {
        return StringEquals( a, b ) == false;
    }

    i32 StringFind( StringView str, StringView find ) {
        if( find.count == 0 ) {
            return 0;
        }
        if( find.count > str.count ) {
            return -1;
        }

        const i32 last = str.count - find.count;
        for( i32 i = 0; i <= last; i++ ) {
            if( memcmp( str.data + i, find.data, (size_t)find.count ) == 0 ) {
                return i;
            }
        }
        return -1;
    }

    i32 StringFindLast( StringView str, StringView find ) {
        if( find.count == 0 ) {
            return str.count;
        }
        if( find.count > str.count ) {
            return -1;
        }

        for( i32 i = str.count - find.count; i >= 0; i-- ) {
            if( memcmp( str.data + i, find.data, (size_t)find.count ) == 0 ) {
                return i;
            }
        }
        return -1;
    }

    i32 StringFindChar( StringView str, char find ) {
        for( i32 i = 0; i < str.count; i++ ) {
            if( str.data[i] == find ) {
                return i;
            }
        }
        return -1;
    }

    i32 StringFindLastChar( StringView str, char find ) {
        for( i32 i = str.count - 1; i >= 0; i-- ) {
            if( str.data[i] == find ) {
                return i;
            }
        }
        return -1;
    }

    bool StringContains( StringView str, StringView find ) {
        return StringFind( str, find ) >= 0;
    }

    bool StringStartsWith( StringView str, StringView prefix ) {
        if( prefix.count > str.count ) {
            return false;
        }
        return StringEquals( StringView( str.data, prefix.count ), prefix );
    }

    bool StringEndsWith( StringView str, StringView suffix ) {
        if( suffix.count > str.count ) {
            return false;
        }
        return StringEquals( StringView( str.data + str.count - suffix.count, suffix.count ),
                             suffix );
    }

    StringView StringSubstring( StringView str, i32 start, i32 count ) {
        if( start < 0 ) {
            start = 0;
        }
        if( start > str.count ) {
            start = str.count;
        }

        i32 available = str.count - start;
        if( count < 0 || count > available ) {
            count = available;
        }
        return StringView( str.data + start, count );
    }

    StringView StringTrimLeft( StringView str ) {
        i32 start = 0;
        while( start < str.count && IsSpace( str.data[start] ) ) {
            start++;
        }
        return StringView( str.data + start, str.count - start );
    }

    StringView StringTrimRight( StringView str ) {
        i32 end = str.count;
        while( end > 0 && IsSpace( str.data[end - 1] ) ) {
            end--;
        }
        return StringView( str.data, end );
    }

    StringView StringTrim( StringView str ) {
        return StringTrimLeft( StringTrimRight( str ) );
    }

    bool StringSplitNext( StringView * cursor, char separator, StringView * outPart ) {
        // count < 0 is the exhausted marker, set once the last field is handed
        // out. Without it a trailing separator could not be told apart from the
        // end of the string.
        if( cursor->count < 0 ) {
            return false;
        }

        const i32 index = StringFindChar( *cursor, separator );
        if( index < 0 ) {
            *outPart = *cursor;
            cursor->count = -1;
            return true;
        }

        *outPart = StringView( cursor->data, index );
        cursor->data += index + 1;
        cursor->count -= index + 1;
        return true;
    }

    i32 StringSplit( StringView str, char separator, StringView * outParts, i32 maxParts ) {
        StringView cursor = str;
        StringView part = {};
        i32 written = 0;

        while( written < maxParts && StringSplitNext( &cursor, separator, &part ) ) {
            outParts[written] = part;
            written++;
        }
        return written;
    }

    bool StringParseI32( StringView str, i32 * outValue ) {
        str = StringTrim( str );
        if( str.count == 0 ) {
            return false;
        }

        i32 index = 0;
        bool negative = false;
        if( str.data[0] == '+' || str.data[0] == '-' ) {
            negative = str.data[0] == '-';
            index = 1;
        }
        if( index >= str.count ) {
            return false;
        }

        // Accumulated in 64 bits so the range check happens before it wraps.
        i64 value = 0;
        for( ; index < str.count; index++ ) {
            const char c = str.data[index];
            if( c < '0' || c > '9' ) {
                return false;
            }

            value = value * 10 + ( c - '0' );
            if( value > 2147483648ll ) {
                return false;
            }
        }

        if( negative ) {
            value = -value;
        }
        if( value > 2147483647ll || value < -2147483648ll ) {
            return false;
        }

        *outValue = (i32)value;
        return true;
    }

    bool StringParseF32( StringView str, f32 * outValue ) {
        str = StringTrim( str );
        if( str.count == 0 ) {
            return false;
        }

        // strtof needs a terminator and a view has none, so it goes through a
        // stack copy. Anything longer than this is not a float.
        char buffer[64] = {};
        if( str.count >= (i32)sizeof( buffer ) ) {
            return false;
        }
        memcpy( buffer, str.data, (size_t)str.count );

        char * end = nullptr;
        const float parsed = strtof( buffer, &end );
        if( end != buffer + str.count ) {
            return false;
        }

        *outValue = (f32)parsed;
        return true;
    }

    StringBuffer StringBufferFrom( HeapString & str ) {
        return StringBuffer{ str.data, &str.count, str.cap };
    }

    void StringBufferClear( StringBuffer buffer ) {
        *buffer.count = 0;
        if( buffer.data != nullptr && buffer.cap > 0 ) {
            buffer.data[0] = '\0';
        }
    }

    bool StringBufferAppend( StringBuffer buffer, StringView value ) {
        if( buffer.data == nullptr || buffer.cap <= 0 ) {
            return value.count == 0;
        }

        // One byte is always held back for the terminator.
        const i32 room = buffer.cap - 1 - *buffer.count;
        i32 copied = value.count;
        bool fitted = true;
        if( copied > room ) {
            copied = room > 0 ? room : 0;
            fitted = false;
        }

        if( copied > 0 ) {
            memcpy( buffer.data + *buffer.count, value.data, (size_t)copied );
            *buffer.count += copied;
        }
        buffer.data[*buffer.count] = '\0';
        return fitted;
    }

    bool StringBufferAppendChar( StringBuffer buffer, char value ) {
        return StringBufferAppend( buffer, StringView( &value, 1 ) );
    }

    bool StringBufferSet( StringBuffer buffer, StringView value ) {
        StringBufferClear( buffer );
        return StringBufferAppend( buffer, value );
    }

    void HeapStringReserve( HeapString & str, i32 wantedCap ) {
        if( wantedCap <= str.cap ) {
            return;
        }

        i32 newCap = str.cap != 0 ? str.cap : 16;
        while( newCap < wantedCap ) {
            newCap *= 2;
        }

        char * newData = (char *)::realloc( str.data, (size_t)newCap );
        if( newData == nullptr ) {
            return;
        }

        str.data = newData;
        str.cap = newCap;
        str.data[str.count] = '\0';
    }

    HeapString HeapStringCreate( StringView value ) {
        HeapString str = {};
        StringAppend( str, value );
        return str;
    }

    void HeapStringFree( HeapString & str ) {
        ::free( str.data );
        str.data = nullptr;
        str.count = 0;
        str.cap = 0;
    }

    void HeapStringClear( HeapString & str ) {
        str.count = 0;
        if( str.data != nullptr && str.cap > 0 ) {
            str.data[0] = '\0';
        }
    }

    bool StringAppend( HeapString & str, StringView value ) {
        if( value.count <= 0 ) {
            return true;
        }

        // Room for the text and the terminator.
        HeapStringReserve( str, str.count + value.count + 1 );
        if( str.cap < str.count + value.count + 1 ) {
            return false;
        }

        memcpy( str.data + str.count, value.data, (size_t)value.count );
        str.count += value.count;
        str.data[str.count] = '\0';
        return true;
    }

    bool StringSet( HeapString & str, StringView value ) {
        HeapStringClear( str );
        return StringAppend( str, value );
    }

} // namespace sol
