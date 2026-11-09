// Runs at the head of `native_world_drawn` (the `world_drawn` render hook point: once the world is drawn, before the
// hand and the HUD, on the render thread): the visual debug mode draws its world layers (sources and their occlusion
// rays, paths round walls, the reflection grid and rays) over the frame. With every layer off it does nothing.
fn steam_audio_world_drawn(world: &mut WorldDrawn<'_>) {
    crate::audio::steam_audio::debug_draw(world);
}
