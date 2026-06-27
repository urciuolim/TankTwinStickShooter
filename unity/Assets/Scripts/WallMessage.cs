using Newtonsoft.Json.Linq;

// WallMessage: builds the ONE-TIME map wall-layout message (Unity -> Python). Additive seam,
// SEPARATE from the per-step RL path (state JSON + optional binary frame + actions). This message
// is NEVER interleaved into SendAndReceiveData(); it is its own discrete nwStream.Write issued by
// DriverController during the pre-game handshake (ReceiveAndSendData), so the 52-float state layout,
// the binary frame contract (FrameCapture), the action messages, and their timing stay byte-for-byte
// unchanged. Unity is the source of truth for the map; Python tracks map-state from this message
// instead of re-parsing the arena JSON.
//
// =====================================================================================
// WALL MESSAGE WIRE CONTRACT (must match the Python core.protocol parser byte-for-byte)
// =====================================================================================
// The socket is UNFRAMED: one JSON object per recv. This message is sent as its OWN discrete
// nwStream.Write, never bundled with a handshake confirmation in the same write. It is strict-JSON
// (built with Newtonsoft JObject/JArray, serialized via .ToString()) so Python's `json` module
// (no trailing commas, no leading-dot floats) parses it directly. All cell coordinates are INTS.
//
// Top-level shape (the "type" tag disambiguates it from every other message Python may recv):
//
//   {
//     "type": "walls",          // string tag, ALWAYS exactly "walls" (disambiguator)
//     "map_id": "<id>",         // string: the arena identity (e.g. the switch_arena path or
//                               //         the configured arena_path). May be empty string "".
//     "tileID": <int>,          // int: the wall tileID from arena["Walls"]["tileID"]
//     "dims": {                 // grid bounds copied verbatim from arena["Walls"]["dims"]
//       "minX": <int>, "maxX": <int>,
//       "minY": <int>, "maxY": <int>
//     },
//     "columns": {              // OCCUPIED wall cells, mirroring the SOURCE shape exactly:
//       "<x>": [y0, y1, ...],   //   one key per column x that HAS occupied cells; the value is
//       ...                     //   the ascending list of occupied y-cells for that column.
//     }                         //   Columns with no occupied cells are OMITTED (not "<x>": []).
//   }
//
// REPRESENTATION CHOICE: we mirror the source per-column occupied-y lists (arena["Walls"]["<x>"])
// rather than emitting explicit [x,y] pairs. This is compact, unambiguous, and lets Python
// reconstruct the exact occupied set as {(x, y) for x in columns for y in columns[x]}. Only
// columns that actually carry occupied cells are emitted (the source itself omits empty columns),
// so a wall-free arena yields "columns": {} -- still valid strict JSON.
//
// SEND ORDERING vs the handshake confirmations (PINNED -- Python reads in this order):
//   (a) initial load  (start handshake): DriverController writes {"starting":true}  FIRST, THEN
//       this walls message as a SEPARATE write.
//   (b) map change   (switch_arena):     DriverController writes {"arena_switched":true} FIRST,
//       THEN this walls message as a SEPARATE write.
// In both cases the confirmation comes first, the walls message second; two discrete writes,
// two discrete Python recvs. If the arena is null (no map configured) NO walls message is sent
// (the confirmation is still sent), so Python must treat the walls message as optional-but-once.
// =====================================================================================
public static class WallMessage
{
    // The fixed top-level disambiguator tag value. Python matches on this exact string.
    public const string TypeTag = "walls";

    // Pure builder: takes the arena["Walls"] JToken (the same shape GameController.PlaceTiles reads)
    // plus the resolved map id, and returns the strict-JSON wall message JObject. NO MonoBehaviour,
    // socket, or Unity-scene dependency, so it is fully unit-testable in an EditMode test. Caller
    // serializes via .ToString() for the wire (the established pattern in this codebase).
    //
    // walls MUST be non-null and carry "dims" + "tileID" (the shape every arena file uses). The
    // null-arena guard lives in DriverController (it skips the send entirely), so this method is
    // only ever called with a real Walls token.
    public static JObject Build(JToken walls, string mapId)
    {
        JToken dimsToken = walls["dims"];
        int minX = dimsToken["minX"].Value<int>();
        int maxX = dimsToken["maxX"].Value<int>();
        int minY = dimsToken["minY"].Value<int>();
        int maxY = dimsToken["maxY"].Value<int>();
        int tileID = walls["tileID"].Value<int>();

        JObject dims = new JObject
        {
            ["minX"] = minX,
            ["maxX"] = maxX,
            ["minY"] = minY,
            ["maxY"] = maxY
        };

        // Mirror the source per-column occupied-y lists. We iterate minX..maxX (the same loop
        // PlaceTiles uses) so the emitted columns/cells match exactly what the game places as
        // wall tiles -- Unity is the source of truth. Columns with no occupied cells are omitted.
        JObject columns = new JObject();
        for (int x = minX; x <= maxX; x++)
        {
            JToken yaxisToken = walls[x.ToString()];
            if (yaxisToken == null)
                continue; // column key absent in source => no occupied cells in this column
            JArray yaxis = (JArray)yaxisToken;
            if (yaxis.Count == 0)
                continue; // explicit empty list => nothing occupied; omit for a tight message

            JArray cells = new JArray();
            foreach (JToken y in yaxis)
                cells.Add(y.Value<int>()); // ints only -- no float concerns

            columns[x.ToString()] = cells;
        }

        JObject message = new JObject
        {
            ["type"] = TypeTag,
            ["map_id"] = mapId ?? string.Empty,
            ["tileID"] = tileID,
            ["dims"] = dims,
            ["columns"] = columns
        };

        return message;
    }
}
