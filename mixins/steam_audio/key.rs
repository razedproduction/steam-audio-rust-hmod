// Runs at the head of `native_key` (the `key` hook point: a key went down or up, on any screen): while the settings
// page's block search is being typed, a key goes into it and the page is asked for its labels again, so the list
// follows the search as it is typed. Back in the game, the search ends.
fn steam_audio_key(game: &mut crate::game::Game<'_>, key: &str, pressed: bool) {
    use horizon_client_render::audio::steam_audio::{
        MOD_ID, settings_key, settings_stop_typing, settings_typing,
    };
    if !pressed || !settings_typing() {
        return;
    }
    if game.native_playing() {
        settings_stop_typing();
        return;
    }
    if settings_key(key)
        && let Some(i) = game.settings.mods.installed.iter().position(|m| m.id == MOD_ID)
    {
        game.native_settings_refresh(i);
    }
}
