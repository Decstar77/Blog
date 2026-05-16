package geomesh_win32

// Desktop entry point. Uses GLFW for windowing/input. Camera input mirrors the
// Unreal-Engine viewport: hold RMB to look + fly with WASD, Q/E for vertical,
// Shift to sprint, and the scroll wheel changes movement speed.

import "core:fmt"
import "core:time"
import glfw "vendor:glfw"
import gl   "vendor:OpenGL"

import logic "../logic"

GL_MAJOR :: 4
GL_MINOR :: 1
MSAA_SAMPLES :: 8

WIN_W :: 1280
WIN_H :: 720

g_app: logic.App

// Mouse capture state. We accumulate raw cursor deltas while RMB is held and
// hand them off to the logic layer once per frame.
g_capturing:        bool
g_first_capture:    bool
g_last_mouse_x:     f64
g_last_mouse_y:     f64
g_pending_dx:       f32
g_pending_dy:       f32
g_pending_scroll:   f32
g_lmb_down:         bool

framebuffer_size_callback :: proc "c" (window: glfw.WindowHandle, width, height: i32) {
    gl.Viewport(0, 0, width, height)
}

mouse_button_callback :: proc "c" (window: glfw.WindowHandle, button, action, mods: i32) {
    if button == glfw.MOUSE_BUTTON_RIGHT {
        if action == glfw.PRESS {
            g_capturing     = true
            g_first_capture = true
            glfw.SetInputMode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        } else if action == glfw.RELEASE {
            g_capturing = false
            glfw.SetInputMode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        }
    } else if button == glfw.MOUSE_BUTTON_LEFT {
        if action == glfw.PRESS   do g_lmb_down = true
        if action == glfw.RELEASE do g_lmb_down = false
    }
}

cursor_pos_callback :: proc "c" (window: glfw.WindowHandle, x, y: f64) {
    if g_capturing {
        if g_first_capture {
            g_last_mouse_x  = x
            g_last_mouse_y  = y
            g_first_capture = false
            return
        }
        g_pending_dx += f32(x - g_last_mouse_x)
        g_pending_dy += f32(y - g_last_mouse_y)
    }
    g_last_mouse_x = x
    g_last_mouse_y = y
}

scroll_callback :: proc "c" (window: glfw.WindowHandle, xoffset, yoffset: f64) {
    g_pending_scroll += f32(yoffset)
}

key_held :: proc(window: glfw.WindowHandle, key: i32) -> bool {
    return glfw.GetKey(window, key) == glfw.PRESS
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
    glfw.WindowHint(glfw.SAMPLES, MSAA_SAMPLES)

    window := glfw.CreateWindow(WIN_W, WIN_H, "GeoMesh", nil, nil)
    if window == nil {
        fmt.eprintln("glfw.CreateWindow failed")
        return
    }
    defer glfw.DestroyWindow(window)

    glfw.MakeContextCurrent(window)
    glfw.SwapInterval(1) // vsync
    glfw.SetFramebufferSizeCallback(window, framebuffer_size_callback)
    glfw.SetMouseButtonCallback(window, mouse_button_callback)
    glfw.SetCursorPosCallback(window, cursor_pos_callback)
    glfw.SetScrollCallback(window, scroll_callback)

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

        fb_w, fb_h = glfw.GetFramebufferSize(window)
        aspect := f32(fb_w) / f32(max(fb_h, 1))

        // Build this frame's input snapshot. Movement keys only matter while
        // we're in look-mode (RMB held), but we still pass them through and
        // let `update_camera` gate on `look_active`.
        g_app.input = logic.Input{
            forward     = key_held(window, glfw.KEY_W),
            back        = key_held(window, glfw.KEY_S),
            left        = key_held(window, glfw.KEY_A),
            right       = key_held(window, glfw.KEY_D),
            up          = key_held(window, glfw.KEY_SPACE),
            down        = key_held(window, glfw.KEY_LEFT_CONTROL),
            boost       = key_held(window, glfw.KEY_LEFT_SHIFT) || key_held(window, glfw.KEY_RIGHT_SHIFT),
            look_active = g_capturing,
            mouse_dx    = g_pending_dx,
            mouse_dy    = g_pending_dy,
            scroll_dy   = g_pending_scroll,
            aspect      = aspect,
            mouse_x     = f32(g_last_mouse_x),
            mouse_y     = f32(g_last_mouse_y),
            viewport_w  = f32(fb_w),
            viewport_h  = f32(fb_h),
            lmb         = g_lmb_down,
            k1          = key_held(window, glfw.KEY_1),
            k2          = key_held(window, glfw.KEY_2),
            k3          = key_held(window, glfw.KEY_3),
        }
        g_pending_dx     = 0
        g_pending_dy     = 0
        g_pending_scroll = 0

        logic.frame(&g_app, dt)
        glfw.SwapBuffers(window)
    }
}
