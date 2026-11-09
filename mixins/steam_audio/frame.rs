// Runs at the head of `native_frame` (the `frame` hook point: once a frame, on the game thread): the first time, the
// support module reads the settings kept from the mod's settings page; every time, it works out the world's
// acoustics around the camera from its blocks, and gives the settings page the game's blocks and block tags.
fn steam_audio_acoustics(game: &mut crate::game::Game<'_>) {
    horizon_client_render::audio::steam_audio::load_settings(&|key| {
        game.native_setting(horizon_client_render::audio::steam_audio::MOD_ID, key)
    });
    // The block tags (for the settings page's blocks section), once the world's resources are there.
    let tags = game.res.rest.get().map(|rest| &rest.items.block_tags);
    horizon_client_render::audio::steam_audio::update_acoustics(
        &game.audio,
        &game.world,
        game.res.facts(),
        tags,
    );
}
