using Silk.NET.OpenGL;

namespace Engine {
    public static class OpenGLInstance {
        private static GL? instance = null;

        public static void  Set(GL gl) { instance = gl; }
        public static GL    Get() { return instance!; }
    }
}
