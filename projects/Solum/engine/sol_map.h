#pragma once
#include "sol_brush.h"
#include "sol_list.h"
#include "sol_string.h"

// A level as authored: a flat list of brushes. Saved as text, one face per
// line, so a map diffs and merges like source code.
//
//   solum_map 1
//   brush
//   face <nx> <ny> <nz> <distance> <offsetU> <offsetV> <scaleU> <scaleV> <rotation> <material...>
//   ...
//   end
//
// Only planes and texture settings are stored. Corners are derived again on
// load, the same way every edit derives them, so a file can never hold a
// polygon that disagrees with its planes. The material runs to the end of the
// line and may be empty.

namespace sol {

    constexpr i32 kMapVersion = 1;

    struct Map {
        List<Brush>     brushes;
    };

    void    MapFree( Map & map );
    Map     MapCopy( const Map & map );

    // Editor flags - selection, hidden - are not saved: a file is the level,
    // not the state of the tool that last had it open.
    bool    MapSave( const Map & map, StringView path );
    // Replaces map's contents. Brushes whose planes no longer enclose anything
    // are skipped with a warning rather than failing the whole load.
    bool    MapLoad( Map & map, StringView path );

    // The same text a file holds, without the file: what the editor puts on
    // the clipboard, so brushes copy between maps and between editors.
    // selectedOnly writes just the brushes flagged selected.
    void    MapWriteText( const Map & map, bool selectedOnly, HeapString & outText );
    // sourceName only labels the warnings.
    bool    MapReadText( Map & map, StringView text, StringView sourceName );

} // namespace sol
