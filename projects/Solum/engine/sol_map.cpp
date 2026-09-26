#include "sol_map.h"
#include "sol_asset.h"

#include <cstdio>

namespace sol {

    void MapFree( Map & map ) {
        for( i32 i = 0; i < map.brushes.count; i++ ) {
            BrushFree( map.brushes[i] );
        }
        ListFree( map.brushes );
    }

    Map MapCopy( const Map & map ) {
        Map copy = {};
        ListReserve( copy.brushes, map.brushes.count );
        for( i32 i = 0; i < map.brushes.count; i++ ) {
            ListAdd( copy.brushes, BrushCopy( map.brushes[i] ) );
        }
        return copy;
    }

    // --- writing ------------------------------------------------------------

    // %.9g round-trips every f32 exactly, so a save and a load never move a
    // plane, however many times a map goes through them.
    static void AppendFloat( HeapString & text, f32 value ) {
        char buffer[32] = {};
        snprintf( buffer, sizeof( buffer ), " %.9g", (double)value );
        StringAppend( text, buffer );
    }

    void MapWriteText( const Map & map, bool selectedOnly, HeapString & text ) {
        StringSet( text, "solum_map 1\n" );

        for( i32 b = 0; b < map.brushes.count; b++ ) {
            const Brush & brush = map.brushes[b];
            if( selectedOnly && !( brush.flags & BrushFlag_Selected ) ) {
                continue;
            }
            StringAppend( text, "brush\n" );
            for( i32 f = 0; f < brush.faces.count; f++ ) {
                const BrushFace & face = brush.faces[f];
                StringAppend( text, "face" );
                AppendFloat( text, face.plane.normal.x );
                AppendFloat( text, face.plane.normal.y );
                AppendFloat( text, face.plane.normal.z );
                AppendFloat( text, face.plane.distance );
                AppendFloat( text, face.texture.offsetU );
                AppendFloat( text, face.texture.offsetV );
                AppendFloat( text, face.texture.scaleU );
                AppendFloat( text, face.texture.scaleV );
                AppendFloat( text, face.texture.rotation );
                StringAppend( text, " " );
                StringAppend( text, face.texture.material );
                StringAppend( text, "\n" );
            }
            StringAppend( text, "end\n" );
        }
    }

    bool MapSave( const Map & map, StringView path ) {
        HeapString text = {};
        MapWriteText( map, false, text );
        const bool ok = FileWriteEntire( path, text.data, text.count );
        HeapStringFree( text );
        return ok;
    }

    // --- reading ------------------------------------------------------------

    // Next run of non-blank characters, advancing the cursor past it.
    static bool NextToken( StringView * cursor, StringView * outToken ) {
        i32 start = 0;
        while( start < cursor->count && ( cursor->data[start] == ' ' || cursor->data[start] == '\t' ) ) {
            start++;
        }
        i32 end = start;
        while( end < cursor->count && cursor->data[end] != ' ' && cursor->data[end] != '\t' ) {
            end++;
        }
        if( end == start ) {
            return false;
        }
        *outToken = StringView( cursor->data + start, end - start );
        *cursor = StringView( cursor->data + end, cursor->count - end );
        return true;
    }

    static bool ParseFace( StringView line, BrushFace * outFace ) {
        f32 values[9] = {};
        for( i32 i = 0; i < 9; i++ ) {
            StringView token = {};
            if( !NextToken( &line, &token ) || !StringParseF32( token, &values[i] ) ) {
                return false;
            }
        }

        BrushFace face = {};
        face.plane.normal = Vec3Normalize( Vec3{ values[0], values[1], values[2] } );
        face.plane.distance = values[3];
        face.texture = FaceTextureDefault();
        face.texture.offsetU = values[4];
        face.texture.offsetV = values[5];
        face.texture.scaleU = values[6];
        face.texture.scaleV = values[7];
        face.texture.rotation = values[8];
        StringSet( face.texture.material, StringTrim( line ) );

        if( Vec3Length( face.plane.normal ) < 0.5f ) {
            return false;
        }
        *outFace = face;
        return true;
    }

    bool MapLoad( Map & map, StringView path ) {
        List<u8> bytes = {};
        if( !FileReadEntire( path, bytes ) ) {
            return false;
        }
        const bool ok = MapReadText( map, StringView( (const char *)bytes.data, bytes.count ), path );
        ListFree( bytes );
        return ok;
    }

    bool MapReadText( Map & map, StringView text, StringView sourceName ) {
        const StringView path = sourceName;
        Map loaded = {};
        Brush current = {};
        bool inBrush = false;
        bool sawHeader = false;
        bool ok = true;
        i32 lineNumber = 0;
        i32 skipped = 0;

        StringView cursor = text;
        StringView line = {};
        while( ok && StringSplitNext( &cursor, '\n', &line ) ) {
            lineNumber++;
            line = StringTrim( line );
            if( line.count == 0 || line.data[0] == '#' ) {
                continue;
            }

            StringView rest = line;
            StringView keyword = {};
            NextToken( &rest, &keyword );

            if( !sawHeader ) {
                // Anything else first is some other file that happens to have
                // the extension, and reading on would only produce noise.
                StringView versionToken = {};
                i32 version = 0;
                if( keyword != "solum_map" || !NextToken( &rest, &versionToken ) ||
                    !StringParseI32( versionToken, &version ) || version > kMapVersion ) {
                    fprintf( stderr, "MapReadText: '%.*s' is not a map this build can read\n", path.count, path.data );
                    ok = false;
                }
                sawHeader = true;
            } else if( keyword == "brush" && !inBrush ) {
                current = {};
                inBrush = true;
            } else if( keyword == "face" && inBrush ) {
                BrushFace face = {};
                if( !ParseFace( rest, &face ) ) {
                    fprintf( stderr, "MapReadText: bad face on line %d\n", lineNumber );
                    ok = false;
                } else {
                    ListAdd( current.faces, face );
                }
            } else if( keyword == "end" && inBrush ) {
                inBrush = false;
                if( BrushRebuild( current ) ) {
                    ListAdd( loaded.brushes, current );
                } else {
                    skipped++;
                    BrushFree( current );
                }
                current = {};
            } else {
                fprintf( stderr, "MapReadText: unexpected '%.*s' on line %d\n", keyword.count, keyword.data, lineNumber );
                ok = false;
            }
        }

        BrushFree( current );

        if( !ok || inBrush ) {
            MapFree( loaded );
            return false;
        }
        if( skipped > 0 ) {
            fprintf( stderr, "MapReadText: skipped %d brushes that enclose no volume\n", skipped );
        }

        MapFree( map );
        map = loaded;
        return true;
    }

} // namespace sol
