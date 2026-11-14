using System.Net;
using System.Text;
using System.Text.Json;

namespace arcade {
    /*
    ===============================================================================
    Generic training dashboard.

    Training code logs scalars by key, the dashboard plots them in the browser.
      TrainingDashboard.Start();                                 // serve http://localhost:5005
      TrainingDashboard.BeginRun( "neat", new { seed, settings } ); // new run, config shown in UI
      TrainingDashboard.Log( step, "champion/win", 0.91 );       // one value
      TrainingDashboard.Log( step, new() { ["fitness/best"] = 9.4, ["fitness/mean"] = 3.1 } );

    Keys are "chart/series": every key with the same prefix is drawn on the same chart,
    so keep series that share a prefix on the same scale.

    Every run is appended to diagnostics/runs/<time>_<name>.jsonl (one JSON object per
    line), so old runs can be compared later. The server is a plain file server over that
    folder plus dashboard.html, which is read from disk on every request so it can be
    edited without rebuilding. `dotnet run -- dash` serves the dashboard on its own.
    ===============================================================================
    */
    /*
    ===============================================================================
    A network snapshot for the dashboard's Network tab. Nodes are placed in columns by
    layer (0 = inputs on the left), kind is "input", "hidden" or "output". Edges reference
    node ids. Works for NEAT genomes and plain MLPs alike.
    ===============================================================================
    */
    public record struct NetNode( int id, string kind, int layer, string? label = null, double bias = 0 );
    public record struct NetEdge( int src, int dst, double weight, bool enabled = true );

    public static class TrainingDashboard {
        public static int       Port = 5005;

        private static readonly object  sync = new object();
        private static StreamWriter?    writer;
        private static HttpListener?    listener;
        private static readonly System.Diagnostics.Stopwatch clock = new System.Diagnostics.Stopwatch();

        public static string    Folder  { get; } = FindFolder();
        public static string    RunsDir => Path.Combine( Folder, "runs" );
        public static string?   RunFile { get; private set; }

        /*
        ==========================
        Starts the HTTP server on a background thread. If the port is taken (another
        training process is already serving) this just keeps logging to file: that
        server reads the same runs folder, so the new run still shows up there.
        ==========================
         */
        public static void Start() {
            if ( listener != null ) return;
            var l = new HttpListener();
            l.Prefixes.Add( $"http://localhost:{Port}/" );
            try {
                l.Start();
            }
            catch ( HttpListenerException e ) {
                Console.WriteLine( $"Dashboard: could not listen on port {Port} ({e.ErrorCode}: {e.Message}). Still logging to {RunsDir}" );
                return;
            }
            listener = l;
            new Thread( Serve ) { IsBackground = true, Name = "TrainingDashboard" }.Start();
            Console.WriteLine( $"Dashboard: http://localhost:{Port}/" );
        }

        /*
        ==========================
        Opens a new run file. Config is any object, serialised as-is into the run header.
        ==========================
         */
        public static void BeginRun( string name, object? config = null ) {
            lock ( sync ) {
                EndRun();
                Directory.CreateDirectory( RunsDir );
                string safe = string.Concat( name.Select( c => char.IsLetterOrDigit( c ) || c == '-' || c == '_' ? c : '_' ) );
                RunFile = Path.Combine( RunsDir, $"{DateTime.Now:yyyyMMdd-HHmmss}_{safe}.jsonl" );
                var stream = new FileStream( RunFile, FileMode.Create, FileAccess.Write, FileShare.ReadWrite );
                writer = new StreamWriter( stream, new UTF8Encoding( false ) ) { AutoFlush = true };
                clock.Restart();
                Write( new { type = "meta", name, started = DateTime.Now.ToString( "o" ), config } );
            }
        }

        public static void EndRun() {
            lock ( sync ) {
                writer?.Dispose();
                writer = null;
            }
        }

        public static void Log( int step, string key, double value ) {
            Log( step, new Dictionary<string, double> { [key] = value } );
        }

        /*
        ==========================
        Logs several scalars for one step as a single line. Non-finite values are dropped
        because JSON can't represent them.
        ==========================
         */
        public static void Log( int step, Dictionary<string, double> values ) {
            lock ( sync ) {
                if ( writer == null ) return;
                var finite = values.Where( kv => double.IsFinite( kv.Value ) ).ToDictionary( kv => kv.Key, kv => kv.Value );
                Write( new { step, t = Math.Round( clock.Elapsed.TotalSeconds, 3 ), v = finite } );
            }
        }

        /*
        ==========================
        Logs a network snapshot. Snapshots are large compared to scalars, so log one when
        the network actually changes (e.g. a new champion), not every step. Info is any
        small object shown next to the drawing.
        ==========================
         */
        public static void LogNetwork( int step, string name, IEnumerable<NetNode> nodes, IEnumerable<NetEdge> edges, object? info = null ) {
            lock ( sync ) {
                if ( writer == null ) return;
                Write( new {
                    type = "network", step, name, t = Math.Round( clock.Elapsed.TotalSeconds, 3 ), info,
                    nodes = nodes.Select( n => new { n.id, n.kind, n.layer, n.label, bias = Math.Round( n.bias, 4 ) } ),
                    edges = edges.Select( e => new { e.src, e.dst, w = Math.Round( e.weight, 4 ), on = e.enabled } ),
                } );
            }
        }

        private static readonly JsonSerializerOptions jsonOptions = new JsonSerializerOptions {
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
        };

        private static void Write( object record ) {
            writer!.WriteLine( JsonSerializer.Serialize( record, jsonOptions ) );
        }

        /*
        ==========================
        Walks up from the build output to the folder holding the .csproj, so runs and
        dashboard.html live in the source tree rather than bin/.
        ==========================
         */
        private static string FindFolder() {
            for ( var dir = new DirectoryInfo( AppContext.BaseDirectory ); dir != null; dir = dir.Parent ) {
                if ( dir.GetFiles( "*.csproj" ).Length > 0 ) return Path.Combine( dir.FullName, "diagnostics" );
            }
            return Path.Combine( AppContext.BaseDirectory, "diagnostics" );
        }

        /*
        ==========================
        GET /                     dashboard.html
        GET /runs                 [{ file, size, modified }]
        GET /runs/<file>?from=N   run file bytes from offset N (the client tails it)
        ==========================
         */
        private static void Serve() {
            while ( listener != null && listener.IsListening ) {
                HttpListenerContext ctx;
                try { ctx = listener.GetContext(); }
                catch { return; }
                try { Handle( ctx ); }
                catch ( Exception e ) {
                    try { Reply( ctx, 500, "text/plain", Encoding.UTF8.GetBytes( e.Message ) ); } catch { }
                }
            }
        }

        private static void Handle( HttpListenerContext ctx ) {
            string path = ctx.Request.Url!.AbsolutePath;

            if ( path == "/" || path == "/index.html" ) {
                Reply( ctx, 200, "text/html; charset=utf-8", File.ReadAllBytes( Path.Combine( Folder, "dashboard.html" ) ) );
                return;
            }

            if ( path == "/runs" ) {
                var runs = Directory.Exists( RunsDir )
                    ? new DirectoryInfo( RunsDir ).GetFiles( "*.jsonl" )
                        .OrderByDescending( f => f.Name )
                        .Select( f => new { file = f.Name, size = f.Length, modified = f.LastWriteTimeUtc.ToString( "o" ) } )
                        .ToArray()
                    : [];
                Reply( ctx, 200, "application/json", JsonSerializer.SerializeToUtf8Bytes( runs ) );
                return;
            }

            if ( path.StartsWith( "/runs/" ) ) {
                string name = Path.GetFileName( Uri.UnescapeDataString( path.Substring( 6 ) ) ); // GetFileName blocks ../ escapes
                string file = Path.Combine( RunsDir, name );
                if ( !name.EndsWith( ".jsonl" ) || !File.Exists( file ) ) {
                    Reply( ctx, 404, "text/plain", Encoding.UTF8.GetBytes( "not found" ) );
                    return;
                }
                long.TryParse( ctx.Request.QueryString["from"], out long from );
                using var stream = new FileStream( file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite );
                from = Math.Clamp( from, 0, stream.Length );
                stream.Seek( from, SeekOrigin.Begin );
                byte[] bytes = new byte[stream.Length - from];
                stream.ReadExactly( bytes );
                Reply( ctx, 200, "application/x-ndjson", bytes );
                return;
            }

            Reply( ctx, 404, "text/plain", Encoding.UTF8.GetBytes( "not found" ) );
        }

        private static void Reply( HttpListenerContext ctx, int status, string contentType, byte[] body ) {
            ctx.Response.StatusCode = status;
            ctx.Response.ContentType = contentType;
            ctx.Response.Headers["Cache-Control"] = "no-store";
            ctx.Response.ContentLength64 = body.Length;
            ctx.Response.OutputStream.Write( body );
            ctx.Response.OutputStream.Close();
        }
    }
}
