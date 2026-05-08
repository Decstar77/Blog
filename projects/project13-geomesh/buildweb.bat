@echo off
odin build web -target:js_wasm32 -out:web/geomesh.wasm %*
