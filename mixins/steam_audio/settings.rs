// Runs at the head of `native_settings` (the `settings` hook point, asked when the page opens and after each click):
// the labels of the mod's settings page, one button per setting with its value.
fn steam_audio_settings(game: &mut crate::game::Game<'_>, mod_id: &str, buttons: &mut Vec<String>) {
    if mod_id == horizon_client_render::audio::steam_audio::MOD_ID {
        horizon_client_render::audio::steam_audio::settings_page(&game.settings.language, buttons);
    }
}
