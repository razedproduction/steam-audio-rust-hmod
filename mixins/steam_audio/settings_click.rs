// Runs at the head of `native_settings_click` (the `settings_click` hook point): a button of the mod's settings page
// was clicked. The setting changes at once and is kept with Horizon's `native_set_setting`.
fn steam_audio_settings_click(game: &mut crate::game::Game<'_>, mod_id: &str, button: usize) {
    use horizon_client_render::audio::steam_audio::{MOD_ID, settings_click};
    if mod_id == MOD_ID {
        settings_click(button, &mut |key, value| {
            game.native_set_setting(MOD_ID, key, value)
        });
    }
}
