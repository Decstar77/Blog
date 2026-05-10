package geomesh_web

// Browser entry point. Mirrors win32/main.odin but targets js_wasm32 + WebGL2.
// odin.js auto-loops by calling our exported `step` each requestAnimationFrame.

import "base:runtime"
import "core:fmt"
import gl "vendor:wasm/WebGL"

import logic "../logic"

CANVAS_ID :: "geomesh-canvas"

g_app: logic.App
g_w, g_h: i32

main :: proc() {
    if !gl.CreateCurrentContextById(CANVAS_ID, gl.DEFAULT_CONTEXT_ATTRIBUTES) {
        fmt.eprintln("WebGL2 context creation failed")
        return
    }

    g_w = gl.DrawingBufferWidth()
    g_h = gl.DrawingBufferHeight()

    renderer_init()

    g_app.renderer = make_renderer()
    g_app.renderer.set_viewport(int(g_w), int(g_h))
    logic.initialize(&g_app)
}

@(export)
step :: proc(dt: f64, ctx: ^runtime.Context) -> bool {
    context = ctx^

    // React to canvas resizes triggered by the page / CSS.
    w := gl.DrawingBufferWidth()
    h := gl.DrawingBufferHeight()
    if w != g_w || h != g_h {
        g_w, g_h = w, h
        g_app.renderer.set_viewport(int(g_w), int(g_h))
    }

    // Feed aspect into the logic input so the camera's projection stays in
    // sync with the canvas. No input wiring on the web yet, so the camera
    // simply holds its initial pose.
    g_app.input = logic.Input{aspect = f32(g_w) / f32(max(g_h, 1))}

    logic.frame(&g_app, f32(dt))
    return true
}
