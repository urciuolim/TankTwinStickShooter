using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace TankTwinStickShooter.EditModeTests
{
    // Pins the WALL MESSAGE WIRE CONTRACT produced by WallMessage.Build (see WallMessage.cs) so the
    // Python core.protocol parser stays in lockstep. The builder is pure (takes an arena["Walls"]
    // JToken + a map id, returns a strict-JSON JObject), so it is fully unit-testable here WITHOUT a
    // live scene or socket. The DriverController emission ordering (confirmation first, walls second;
    // null-arena guard) is exercised via the live PlayMode/protocol verify, not here.
    //
    // Contract under test:
    //   { "type":"walls", "map_id":<str>, "tileID":<int>,
    //     "dims":{minX,maxX,minY,maxY}, "columns": { "<x>":[y0,y1,...], ... } }
    //   columns mirror the SOURCE per-column occupied-y lists; empty columns are omitted.
    public class WallMessageTests
    {
        // A small synthetic Walls token shaped EXACTLY like arena["Walls"] (dims + tileID + per-
        // column occupied-y lists). Columns -1 and 1 are occupied; column 0 is deliberately ABSENT
        // (no occupied cells), so we can assert empty columns are omitted from the message.
        private static JToken MakeWalls()
        {
            return new JObject
            {
                ["dims"] = new JObject
                {
                    ["minX"] = -1,
                    ["maxX"] = 1,
                    ["minY"] = -2,
                    ["maxY"] = 2
                },
                ["tileID"] = 1,
                ["-1"] = new JArray(-2, -1, 0),
                ["1"] = new JArray(1, 2)
                // column "0" intentionally omitted (no occupied cells)
            };
        }

        [Test]
        public void Build_TopLevelTypeTagIsWalls()
        {
            JObject msg = WallMessage.Build(MakeWalls(), "Arenas/custom1.json");
            Assert.AreEqual("walls", (string)msg["type"]);
            Assert.AreEqual(WallMessage.TypeTag, (string)msg["type"]);
        }

        [Test]
        public void Build_CarriesMapIdVerbatim()
        {
            JObject msg = WallMessage.Build(MakeWalls(), "Arenas/custom1.json");
            Assert.AreEqual("Arenas/custom1.json", (string)msg["map_id"]);
        }

        [Test]
        public void Build_NullMapIdBecomesEmptyString()
        {
            // Defensive: a null map id must serialize as "" (valid strict JSON string), never JSON null.
            JObject msg = WallMessage.Build(MakeWalls(), null);
            Assert.AreEqual(JTokenType.String, msg["map_id"].Type);
            Assert.AreEqual("", (string)msg["map_id"]);
        }

        [Test]
        public void Build_DimsCopiedVerbatim()
        {
            JObject msg = WallMessage.Build(MakeWalls(), "m");
            JToken dims = msg["dims"];
            Assert.AreEqual(-1, (int)dims["minX"]);
            Assert.AreEqual(1, (int)dims["maxX"]);
            Assert.AreEqual(-2, (int)dims["minY"]);
            Assert.AreEqual(2, (int)dims["maxY"]);
        }

        [Test]
        public void Build_TileIdCarried()
        {
            JObject msg = WallMessage.Build(MakeWalls(), "m");
            Assert.AreEqual(JTokenType.Integer, msg["tileID"].Type);
            Assert.AreEqual(1, (int)msg["tileID"]);
        }

        [Test]
        public void Build_OccupiedColumnsRoundTrip()
        {
            JObject msg = WallMessage.Build(MakeWalls(), "m");
            JObject columns = (JObject)msg["columns"];

            // Column -1: [-2,-1,0]; column 1: [1,2]; column 0 omitted.
            CollectionAssert.AreEqual(new[] { -2, -1, 0 }, columns["-1"].ToObject<int[]>());
            CollectionAssert.AreEqual(new[] { 1, 2 }, columns["1"].ToObject<int[]>());
            Assert.IsNull(columns["0"], "column 0 has no occupied cells and must be omitted");

            // Exactly the two occupied columns are present.
            var keys = new List<string>();
            foreach (var p in columns)
                keys.Add(p.Key);
            CollectionAssert.AreEquivalent(new[] { "-1", "1" }, keys);
        }

        [Test]
        public void Build_CellsAreIntegersNotFloats()
        {
            // Strict-JSON / Python concern: wall cells must be JSON integers (no leading-dot floats).
            JObject msg = WallMessage.Build(MakeWalls(), "m");
            JArray col = (JArray)((JObject)msg["columns"])["-1"];
            foreach (JToken cell in col)
                Assert.AreEqual(JTokenType.Integer, cell.Type, "wall cell must be a JSON integer");
        }

        [Test]
        public void Build_OutputIsStrictJsonParseable()
        {
            // Serialize via .ToString() (the wire form) and parse it back -- proves the on-wire bytes
            // are valid JSON with the full expected structure intact (what Python's json.loads sees).
            string wire = WallMessage.Build(MakeWalls(), "Arenas/custom1.json").ToString();
            JObject roundTripped = JObject.Parse(wire);

            Assert.AreEqual("walls", (string)roundTripped["type"]);
            Assert.AreEqual("Arenas/custom1.json", (string)roundTripped["map_id"]);
            Assert.AreEqual(1, (int)roundTripped["tileID"]);
            Assert.IsNotNull(roundTripped["dims"]);
            Assert.IsNotNull(roundTripped["columns"]);
            CollectionAssert.AreEqual(
                new[] { -2, -1, 0 },
                ((JObject)roundTripped["columns"])["-1"].ToObject<int[]>());
        }

        [Test]
        public void Build_WireHasNoTrailingCommasOrLeadingDotFloats()
        {
            // Newtonsoft never emits trailing commas / leading-dot floats, but pin it explicitly
            // since Python's json is strict. Assert the serialized text is clean.
            string wire = WallMessage.Build(MakeWalls(), "m").ToString();
            StringAssert.Contains("\"type\"", wire);
            Assert.IsFalse(wire.Contains(",}"), "no trailing comma before }");
            Assert.IsFalse(wire.Contains(",]"), "no trailing comma before ]");
            Assert.IsFalse(wire.Contains(": ."), "no leading-dot float after a key");
            Assert.IsFalse(wire.Contains("[."), "no leading-dot float in an array");
        }

        [Test]
        public void Build_EmptyWallLayout_YieldsEmptyColumnsObject()
        {
            // Edge: an arena with a Walls block but NO occupied columns => "columns": {} (valid JSON).
            JToken walls = new JObject
            {
                ["dims"] = new JObject
                {
                    ["minX"] = 0,
                    ["maxX"] = 3,
                    ["minY"] = 0,
                    ["maxY"] = 3
                },
                ["tileID"] = 1
                // no per-column keys at all
            };

            JObject msg = WallMessage.Build(walls, "empty");
            JObject columns = (JObject)msg["columns"];
            Assert.AreEqual(0, columns.Count, "no occupied columns => empty columns object");

            // Still strict-JSON parseable.
            JObject roundTripped = JObject.Parse(msg.ToString());
            Assert.AreEqual(0, ((JObject)roundTripped["columns"]).Count);
        }

        [Test]
        public void Build_ExplicitEmptyColumnList_IsOmitted()
        {
            // Edge: a column present in source but with an EMPTY occupied-y list must be omitted
            // (we never emit "<x>": []), keeping the message tight and the occupied set unambiguous.
            JToken walls = new JObject
            {
                ["dims"] = new JObject
                {
                    ["minX"] = 0,
                    ["maxX"] = 1,
                    ["minY"] = 0,
                    ["maxY"] = 1
                },
                ["tileID"] = 1,
                ["0"] = new JArray(),  // explicit empty -> omitted
                ["1"] = new JArray(0)  // occupied -> kept
            };

            JObject columns = (JObject)WallMessage.Build(walls, "m")["columns"];
            Assert.IsNull(columns["0"], "explicit empty column list must be omitted");
            CollectionAssert.AreEqual(new[] { 0 }, columns["1"].ToObject<int[]>());
        }

        [Test]
        public void Build_RealisticArena_AllSourceColumnsPreserved()
        {
            // A larger fixture resembling the real custom1.json Walls block: a full border column and
            // sparse interior columns. Verifies every occupied source column survives with exact cells.
            JToken walls = new JObject
            {
                ["dims"] = new JObject
                {
                    ["minX"] = -2,
                    ["maxX"] = 2,
                    ["minY"] = -1,
                    ["maxY"] = 1
                },
                ["tileID"] = 1,
                ["-2"] = new JArray(-1, 0, 1), // full column
                ["-1"] = new JArray(1),
                ["0"] = new JArray(-1, 1),
                ["1"] = new JArray(-1),
                ["2"] = new JArray(-1, 0, 1)  // full column
            };

            JObject columns = (JObject)WallMessage.Build(walls, "m")["columns"];
            CollectionAssert.AreEqual(new[] { -1, 0, 1 }, columns["-2"].ToObject<int[]>());
            CollectionAssert.AreEqual(new[] { 1 }, columns["-1"].ToObject<int[]>());
            CollectionAssert.AreEqual(new[] { -1, 1 }, columns["0"].ToObject<int[]>());
            CollectionAssert.AreEqual(new[] { -1 }, columns["1"].ToObject<int[]>());
            CollectionAssert.AreEqual(new[] { -1, 0, 1 }, columns["2"].ToObject<int[]>());
            Assert.AreEqual(5, columns.Count);
        }
    }
}
