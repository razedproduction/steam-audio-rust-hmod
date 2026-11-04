// Runs at the head of `native_frame` (the `frame` hook point: once a frame, on the game thread): the support module
// works out the world's acoustics around the camera from its blocks.
fn steam_audio_acoustics(game: &mut crate::game::Game<'_>) {
    horizon_client_render::audio::steam_audio::update_acoustics(
        &game.audio,
        &game.world,
        game.res.facts(),
    );
}
