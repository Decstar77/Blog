package geomesh_win32

import "base:runtime"
import "core:fmt"
import "core:time"
import win "core:sys/windows"
import gl  "vendor:OpenGL"

import logic "../logic"

GL_MAJOR :: 3
GL_MINOR :: 3

WIN_W :: 1280
WIN_H :: 720

g_running    : bool = true
g_hdc        : win.HDC
g_hglrc      : win.HGLRC
g_app        : logic.App
g_win_w      : i32 = WIN_W
g_win_h      : i32 = WIN_H

g_default_ctx: runtime.Context

window_proc :: proc "system" (hwnd: win.HWND, msg: win.UINT, wparam: win.WPARAM, lparam: win.LPARAM) -> win.LRESULT {
    context = g_default_ctx
    switch msg {
    case win.WM_CLOSE, win.WM_DESTROY:
        g_running = false
        win.PostQuitMessage(0)
        return 0
    case win.WM_SIZE:
        g_win_w = i32(win.LOWORD(cast(win.DWORD)lparam))
        g_win_h = i32(win.HIWORD(cast(win.DWORD)lparam))
        if g_app.renderer.set_viewport != nil && g_win_w > 0 && g_win_h > 0 {
            g_app.renderer.set_viewport(int(g_win_w), int(g_win_h))
        }
        return 0
    }
    return win.DefWindowProcW(hwnd, msg, wparam, lparam)
}

create_gl_context :: proc(hwnd: win.HWND) {
    g_hdc = win.GetDC(hwnd)

    pfd := win.PIXELFORMATDESCRIPTOR{
        nSize       = size_of(win.PIXELFORMATDESCRIPTOR),
        nVersion    = 1,
        dwFlags     = win.PFD_DRAW_TO_WINDOW | win.PFD_SUPPORT_OPENGL | win.PFD_DOUBLEBUFFER,
        iPixelType  = win.PFD_TYPE_RGBA,
        cColorBits  = 32,
        cDepthBits  = 24,
        cStencilBits= 8,
        iLayerType  = win.PFD_MAIN_PLANE,
    }
    pf := win.ChoosePixelFormat(g_hdc, &pfd)
    win.SetPixelFormat(g_hdc, pf, &pfd)

    // Bootstrap a legacy context so we can resolve wglCreateContextAttribsARB.
    dummy := win.wglCreateContext(g_hdc)
    win.wglMakeCurrent(g_hdc, dummy)

    win.wglCreateContextAttribsARB = cast(win.CreateContextAttribsARBType)win.wglGetProcAddress("wglCreateContextAttribsARB")
    win.wglSwapIntervalEXT         = cast(win.SwapIntervalEXTType)win.wglGetProcAddress("wglSwapIntervalEXT")

    if win.wglCreateContextAttribsARB != nil {
        attribs := [?]i32{
            win.CONTEXT_MAJOR_VERSION_ARB, GL_MAJOR,
            win.CONTEXT_MINOR_VERSION_ARB, GL_MINOR,
            win.CONTEXT_PROFILE_MASK_ARB,  win.CONTEXT_CORE_PROFILE_BIT_ARB,
            0,
        }
        g_hglrc = win.wglCreateContextAttribsARB(g_hdc, nil, raw_data(attribs[:]))
        win.wglMakeCurrent(g_hdc, nil)
        win.wglDeleteContext(dummy)
        win.wglMakeCurrent(g_hdc, g_hglrc)
    } else {
        g_hglrc = dummy
    }

    gl.load_up_to(GL_MAJOR, GL_MINOR, win.gl_set_proc_address)

    if win.wglSwapIntervalEXT != nil {
        win.wglSwapIntervalEXT(1) // vsync
    }
}

main :: proc() {
    g_default_ctx = context

    instance := win.HINSTANCE(win.GetModuleHandleW(nil))
    class_name := win.wstring(win.L("GeoMeshWnd"))

    wc := win.WNDCLASSW{
        style         = win.CS_OWNDC | win.CS_HREDRAW | win.CS_VREDRAW,
        lpfnWndProc   = window_proc,
        hInstance     = instance,
        lpszClassName = class_name,
        hCursor       = win.LoadCursorA(nil, win.IDC_ARROW),
    }
    win.RegisterClassW(&wc)

    style := win.WS_OVERLAPPEDWINDOW | win.WS_VISIBLE
    rect := win.RECT{0, 0, WIN_W, WIN_H}
    win.AdjustWindowRect(&rect, style, false)

    hwnd := win.CreateWindowExW(
        0, class_name, win.wstring(win.L("GeoMesh")),
        style,
        win.CW_USEDEFAULT, win.CW_USEDEFAULT,
        rect.right - rect.left, rect.bottom - rect.top,
        nil, nil, instance, nil,
    )
    if hwnd == nil {
        fmt.eprintln("CreateWindowExW failed")
        return
    }

    create_gl_context(hwnd)
    renderer_init()

    g_app.renderer = make_renderer()
    g_app.renderer.set_viewport(int(g_win_w), int(g_win_h))
    logic.initialize(&g_app)

    last := time.now()
    msg: win.MSG

    for g_running {
        for win.PeekMessageW(&msg, nil, 0, 0, win.PM_REMOVE) {
            if msg.message == win.WM_QUIT {
                g_running = false
                break
            }
            win.TranslateMessage(&msg)
            win.DispatchMessageW(&msg)
        }
        if !g_running do break

        now := time.now()
        dt  := f32(time.duration_seconds(time.diff(last, now)))
        last = now

        logic.frame(&g_app, dt)
        renderer_flush()
        win.SwapBuffers(g_hdc)
    }
}
