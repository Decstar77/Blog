#pragma once
#include "sol_defines.h"

namespace sol {

    // The common currency. Every read-only string function takes one of these by
    // value, and every string type below converts to it implicitly, so there is
    // exactly one overload of each function rather than one per string type.
    //
    // A view does not own or null-terminate anything - it is a borrowed span, and
    // it is only valid while whatever it points at is.
    //
    // This is the one type in the engine with constructors. That is deliberate:
    // its whole job is being converted into, and the implicit constructor from
    // const char * is what makes string literals work at call sites.
    struct StringView {
        const char *    data;
        i32             count;

        StringView() = default;
        StringView( const char * cstr );                    // counts to the null terminator
        StringView( const char * chars, i32 charCount );

        char            operator[]( i32 index ) const   { return data[index]; }
        const char *    begin() const                   { return data; }
        const char *    end() const                     { return data + count; }
    };

    // Inline storage, no allocation. _cap_ counts the null terminator, so the most characters this can hold is _cap_ - 1.
    template<i32 _cap_>
    struct FixedString {
        char    data[_cap_];
        i32     count;

        operator StringView() const { return StringView( data, count ); }
    };

    using SmallString = FixedString<64>;
    using LargeString = FixedString<256>;

    // Heap allocated and growable. Also kept null-terminated.
    struct HeapString {
        char *  data;
        i32     count;
        i32     cap;

        operator StringView() const { return StringView( data, count ); }
    };

    // The write-side counterpart to StringView: a borrowed, fixed-capacity
    // buffer. count is a pointer so appends update the string that owns it.
    //
    // This is what lets one StringBufferAppend serve every string type. It can
    // fill a buffer but never grow one, because it cannot realloc through a
    // borrowed pointer - growing is HeapString's own business.
    struct StringBuffer {
        char *  data;
        i32 *   count;
        i32     cap;    // includes the null terminator
    };

    // ---- reads -------------------------------------------------------------

    i32         StringLength( const char * cstr );
    bool        StringIsEmpty( StringView str );

    bool        StringEquals( StringView a, StringView b );
    bool        StringEqualsIgnoreCase( StringView a, StringView b );
    i32         StringCompare( StringView a, StringView b ); // Lexicographic: negative, zero or positive, like strcmp.

    bool        operator==( StringView a, StringView b );
    bool        operator!=( StringView a, StringView b );

    // Index of the first/last match, or -1. An empty needle matches at 0.
    i32         StringFind( StringView str, StringView find );
    i32         StringFindLast( StringView str, StringView find );
    i32         StringFindChar( StringView str, char find );
    i32         StringFindLastChar( StringView str, char find );
    bool        StringContains( StringView str, StringView find );

    bool        StringStartsWith( StringView str, StringView prefix );
    bool        StringEndsWith( StringView str, StringView suffix );

    // Clamped to the bounds of str rather than failing.
    StringView  StringSubstring( StringView str, i32 start, i32 count );

    StringView  StringTrimLeft( StringView str );
    StringView  StringTrimRight( StringView str );
    StringView  StringTrim( StringView str );

    // Walks separator-delimited fields without allocating. Seed the cursor with the whole view and call until it returns false. Empty fields come back as
    // empty views, so "a,,b" yields three parts.
    bool        StringSplitNext( StringView * cursor, char separator, StringView * outPart );
    // Fills outParts and returns how many were written, stopping at maxParts.
    i32         StringSplit( StringView str, char separator, StringView * outParts, i32 maxParts );

    // False if the text is not entirely a number, so a partial parse never passes silently.
    bool        StringParseI32( StringView str, i32 * outValue );
    bool        StringParseF32( StringView str, f32 * outValue );

    // ---- writes ------------------------------------------------------------

    template<i32 _cap_> 
    StringBuffer StringBufferFrom( FixedString<_cap_> & str ) { return StringBuffer{ str.data, &str.count, _cap_ }; }
    StringBuffer StringBufferFrom( HeapString & str );

    void        StringBufferClear( StringBuffer buffer );
    bool        StringBufferAppend( StringBuffer buffer, StringView value );
    bool        StringBufferAppendChar( StringBuffer buffer, char value );
    bool        StringBufferSet( StringBuffer buffer, StringView value );

    template<i32 _cap_> bool StringAppend( FixedString<_cap_> & str, StringView value ) { return StringBufferAppend( StringBufferFrom( str ), value ); }
    template<i32 _cap_> bool StringSet( FixedString<_cap_> & str, StringView value ) { return StringBufferSet( StringBufferFrom( str ), value ); }
    template<i32 _cap_> void StringClear( FixedString<_cap_> & str ) { StringBufferClear( StringBufferFrom( str ) ); }

    // ---- heap strings ------------------------------------------------------

    HeapString  HeapStringCreate( StringView value );
    void        HeapStringFree( HeapString & str );
    void        HeapStringClear( HeapString & str );
    void        HeapStringReserve( HeapString & str, i32 wantedCap );
    bool        StringAppend( HeapString & str, StringView value );
    bool        StringSet( HeapString & str, StringView value );

} // namespace sol
