#pragma once

#include "gs_splats.h"

// Loads a binary little-endian 3D Gaussian Splatting .ply into the scene, replacing whatever was there.
//
// flip_to_y_up rotates the scene 180 degrees about X. Captures reconstructed through COLMAP come out
// +Y down / +Z forward, which is upside down for our +Y up / -Z forward camera.
bool ply_load_scene( const char * path, Scene * scene, bool flip_to_y_up );
