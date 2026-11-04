// Runs at the head of `native_frame` (the `frame` hook point: once a frame, on the game thread): the first time, the
// support module reads the settings kept from the mod's settings page; every time, it works out the world's
// acoustics around the camera from its blocks.
fn steam_audio_acoustics(game: &mut crate::game::Game<'_>) {
    horizon_client_render::audio::steam_audio::load_settings(&|key| {
        game.native_setting(horizon_client_render::audio::steam_audio::MOD_ID, key)
    });
    horizon_client_render::audio::steam_audio::update_acoustics(
        &game.audio,
        &game.world,
        game.res.facts(),
    );
}
