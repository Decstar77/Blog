using Raylib_cs;
using System.Numerics;

namespace arcade {
    // Plays CSLite's events from the point of view of one listener player.
    // Requires Raylib.InitAudioDevice() to have been called.
    public class CSAudio {
        private const string GunPack = "Other/gunsoundpack/Gamemaster Audio - Gun Sound Pack/Gun/";
        private const string Bonus   = "Other/scifisoundsandscifiweapons/Gamemaster Audio - Sci-Fi Sounds and Sci-Fi Weapons/z_ProSoundCollection_BonusSounds/";
        private const string Steps   = "Other/footstepssounds/Footsteps Sounds/WAV/";
        private const string Misc    = "Other/interfaceanditemsounds/V1.0/MISC/";

        private readonly SoundBank Shot = new SoundBank( volume: 0.45f, range: 2000,
            GunPack + "gun_pistol_shot_01.wav",
            GunPack + "gun_pistol_shot_02.wav",
            GunPack + "gun_pistol_shot_03.wav",
            GunPack + "gun_pistol_shot_04.wav",
            GunPack + "gun_pistol_shot_05.wav" );

        // Short range so footsteps give away a nearby enemy but not one across the map.
        private readonly SoundBank Footstep = new SoundBank( volume: 0.5f, range: 450,
            Steps + "Footsteps_Casual_Concrete_01.wav",
            Steps + "Footsteps_Casual_Concrete_02.wav",
            Steps + "Footsteps_Casual_Concrete_03.wav",
            Steps + "Footsteps_Casual_Concrete_04.wav",
            Steps + "Footsteps_Casual_Concrete_05.wav",
            Steps + "Footsteps_Casual_Concrete_06.wav",
            Steps + "Footsteps_Casual_Concrete_07.wav",
            Steps + "Footsteps_Casual_Concrete_08.wav" );

        private readonly SoundBank FleshImpact = new SoundBank( volume: 0.7f, range: 1500,
            Bonus + "bullet_impact_body_flesh_01.wav",
            Bonus + "bullet_impact_body_flesh_02.wav",
            Bonus + "bullet_impact_body_flesh_03.wav" );

        private readonly SoundBank Hurt = new SoundBank( volume: 0.6f, range: 1500,
            Bonus + "voice_male_b_hurt_pain_set_1_04.wav",
            Bonus + "voice_male_b_hurt_pain_set_1_05.wav",
            Bonus + "voice_male_c_hurt_pain_06.wav",
            Bonus + "voice_male_c_hurt_pain_07.wav" );

        private readonly SoundBank Death = new SoundBank( volume: 0.8f, range: 2000,
            Bonus + "voice_male_c_death_04.wav",
            Bonus + "voice_male_c_death_08.wav",
            Bonus + "voice_male_d_death_04.wav" );

        private readonly SoundBank WallHit = new SoundBank( volume: 0.25f, range: 800,
            Bonus + "metal_impact_light_01.wav",
            Bonus + "metal_impact_light_02.wav",
            Bonus + "metal_impact_light_04.wav",
            Bonus + "metal_impact_light_05.wav" );

        private readonly SoundBank RoundWin  = new SoundBank( volume: 0.6f, range: 0, Misc + "You Win (1).wav" );
        private readonly SoundBank RoundLose = new SoundBank( volume: 0.6f, range: 0, Misc + "You Lose (1).wav" );

        public int ListenerId;

        public CSAudio( int listenerId ) {
            ListenerId = listenerId;
        }

        public void Play( CSLite game ) {
            CSPlayer listener = game.Players[ListenerId];

            foreach ( CSEvent e in game.Events ) {
                switch ( e.Type ) {
                    case CSEventType.Shot:     Shot.Play( listener.Position, e.Position ); break;
                    case CSEventType.Footstep: Footstep.Play( listener.Position, e.Position, e.PlayerId == ListenerId ? 0.6f : 1f ); break;
                    case CSEventType.WallHit:  WallHit.Play( listener.Position, e.Position ); break;
                    case CSEventType.Hit:
                        FleshImpact.Play( listener.Position, e.Position );
                        Hurt.Play( listener.Position, e.Position );
                        break;
                    case CSEventType.Death:
                        FleshImpact.Play( listener.Position, e.Position );
                        Death.Play( listener.Position, e.Position );
                        break;
                    case CSEventType.RoundEnd:
                        bool won = game.Result == ( listener.Team == 0 ? CSGameResult.WIN : CSGameResult.LOSS );
                        ( won ? RoundWin : RoundLose ).Play( listener.Position, listener.Position );
                        break;
                }
            }
        }

        public void Unload() {
            foreach ( var bank in new[] { Shot, Footstep, FleshImpact, Hurt, Death, WallHit, RoundWin, RoundLose } ) {
                bank.Unload();
            }
        }

        // A set of interchangeable variations of one sound. Each variation has a few aliases
        // so rapid repeats overlap instead of cutting each other off.
        private class SoundBank {
            private const int Voices = 4;
            private const float PanWidth = 500; // Horizontal offset (world units) that pans fully to one side

            private readonly List<(Sound Source, Sound[] Aliases)> Variants = new();
            private readonly float Volume;
            private readonly float Range; // 0 = not spatial
            private int NextVoice = 0;

            public SoundBank( float volume, float range, params string[] assetNames ) {
                Volume = volume;
                Range = range;
                foreach ( string name in assetNames ) {
                    string path = Assets.AudioPath( name );
                    if ( !File.Exists( path ) ) {
                        Console.WriteLine( $"[CSAudio] Missing sound: {path}" );
                        continue;
                    }
                    Sound source = Raylib.LoadSound( path );
                    var aliases = new Sound[Voices];
                    for ( int v = 0; v < Voices; v++ ) aliases[v] = Raylib.LoadSoundAlias( source );
                    Variants.Add( ( source, aliases ) );
                }
            }

            public void Play( Vector2 listener, Vector2 source, float volumeScale = 1 ) {
                if ( Variants.Count == 0 ) return;

                float volume = Volume * volumeScale;
                float pan = 0;
                if ( Range > 0 ) {
                    float dist = Vector2.Distance( listener, source );
                    float falloff = 1 - Math.Clamp( dist / Range, 0, 1 );
                    if ( falloff <= 0 ) return;
                    volume *= falloff * falloff;
                    pan = Math.Clamp( ( source.X - listener.X ) / PanWidth, -1, 1 ) * 0.8f;
                }

                var (_, aliases) = Variants[Random.Shared.Next( Variants.Count )];
                Sound s = aliases[NextVoice];
                NextVoice = ( NextVoice + 1 ) % Voices;

                Raylib.SetSoundVolume( s, volume );
                Raylib.SetSoundPan( s, pan );
                Raylib.SetSoundPitch( s, 0.94f + (float)Random.Shared.NextDouble() * 0.12f );
                Raylib.PlaySound( s );
            }

            public void Unload() {
                foreach ( var (source, aliases) in Variants ) {
                    foreach ( var a in aliases ) Raylib.UnloadSoundAlias( a );
                    Raylib.UnloadSound( source );
                }
                Variants.Clear();
            }
        }
    }
}
