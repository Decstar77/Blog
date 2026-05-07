package geomesh_win32

// Desktop entry point. Uses GLFW for windowing/input so the same file should
// build on Linux/macOS once the renderer's GL feature set lines up.

import "core:fmt"
import "core:time"
import glfw "vendor:glfw"
import gl   "vendor:OpenGL"

import logic "../logic"

GL_MAJOR :: 3
GL_MINOR :: 3
MSAA_SAMPLES :: 8

WIN_W :: 1280
WIN_H :: 720

g_app: logic.App

framebuffer_size_callback :: proc "c" (window: glfw.WindowHandle, width, height: i32) {
    // No Odin context here; use the contextless GL viewport call directly.
    gl.Viewport(0, 0, width, height)
}

main :: proc() {
    if !glfw.Init() {
        fmt.eprintln("glfw.Init failed")
        return
    }
    defer glfw.Terminate()

    glfw.WindowHint(glfw.CONTEXT_VERSION_MAJOR, GL_MAJOR)
    glfw.WindowHint(glfw.CONTEXT_VERSION_MINOR, GL_MINOR)
    glfw.WindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.WindowHint(glfw.OPENGL_FORWARD_COMPAT, 1)
    glfw.WindowHint(glfw.SAMPLES, MSAA_SAMPLES) // 8x MSAA framebuffer

    window := glfw.CreateWindow(WIN_W, WIN_H, "GeoMesh", nil, nil)
    if window == nil {
        fmt.eprintln("glfw.CreateWindow failed")
        return
    }
    defer glfw.DestroyWindow(window)

    glfw.MakeContextCurrent(window)
    glfw.SwapInterval(1) // vsync
    glfw.SetFramebufferSizeCallback(window, framebuffer_size_callback)

    gl.load_up_to(GL_MAJOR, GL_MINOR, glfw.gl_set_proc_address)
    gl.Enable(gl.MULTISAMPLE)

    renderer_init()

    g_app.renderer = make_renderer()
    fb_w, fb_h := glfw.GetFramebufferSize(window)
    g_app.renderer.set_viewport(int(fb_w), int(fb_h))
    logic.initialize(&g_app)

    last := time.now()
    for !glfw.WindowShouldClose(window) {
        glfw.PollEvents()

        now := time.now()
        dt  := f32(time.duration_seconds(time.diff(last, now)))
        last = now

        logic.frame(&g_app, dt)
        glfw.SwapBuffers(window)
    }
}
