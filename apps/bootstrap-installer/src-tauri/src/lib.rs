use flate2::read::GzDecoder;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{self, Cursor, Read},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tauri::{AppHandle, Emitter, Manager};
use tempfile::TempDir;

const REPOSITORY: &str = "Complexity-ML/labo-ai";
const NODE_VERSION: &str = "v22.14.0";
static INSTALL_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

struct InstallGuard;

fn background_command<S: AsRef<std::ffi::OsStr>>(program: S) -> Command {
    #[allow(unused_mut)]
    let mut command = Command::new(program);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

impl Drop for InstallGuard {
    fn drop(&mut self) {
        INSTALL_IN_PROGRESS.store(false, Ordering::Release);
    }
}

fn begin_install() -> Result<InstallGuard, String> {
    INSTALL_IN_PROGRESS
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map(|_| InstallGuard)
        .map_err(|_| {
            "LABO AI Setup is already installing. Keep this window open to follow its progress."
                .to_string()
        })
}

#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    tarball_url: String,
    assets: Vec<GitHubAsset>,
    #[serde(skip)]
    revision: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GitHubCommit {
    sha: String,
}

#[derive(Debug, Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct InstallState {
    installed_tag: Option<String>,
    installed_channel: Option<String>,
    installed_revision: Option<String>,
    installed_at: Option<u64>,
    app_path: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SetupStatus {
    installed_tag: Option<String>,
    installed_channel: Option<String>,
    installed_revision: Option<String>,
    latest_tag: Option<String>,
    latest_revision: Option<String>,
    channel: String,
    app_path: String,
    platform: &'static str,
    setup_version: &'static str,
    installing: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct InstallResult {
    tag: String,
    path: String,
    setup_relaunched: bool,
}

#[derive(Clone, Serialize)]
struct ProgressEvent {
    stage: String,
    message: String,
    percent: u8,
}

fn client() -> Result<Client, String> {
    Client::builder()
        .user_agent("LABO-AI-Setup")
        .build()
        .map_err(|error| error.to_string())
}

fn latest_release() -> Result<GitHubRelease, String> {
    let api_result = (|| {
        client()?
            .get(format!(
                "https://api.github.com/repos/{REPOSITORY}/releases/latest"
            ))
            .send()
            .and_then(|response| response.error_for_status())
            .map_err(|error| format!("Unable to check GitHub: {error}"))?
            .json::<GitHubRelease>()
            .map_err(|error| format!("Invalid GitHub release response: {error}"))
    })();
    api_result.or_else(|api_error| {
        let response = client()?
            .get(format!("https://github.com/{REPOSITORY}/releases/latest"))
            .send()
            .and_then(|response| response.error_for_status())
            .map_err(|fallback_error| {
                format!("Unable to check GitHub: {api_error}; fallback failed: {fallback_error}")
            })?;
        let tag_name = response
            .url()
            .path_segments()
            .and_then(|segments| segments.last())
            .filter(|tag| tag.starts_with('v'))
            .ok_or_else(|| format!("Unable to resolve the latest GitHub tag after: {api_error}"))?
            .to_string();
        let asset_names = [
            "LABO-AI-Setup-arm64-helper",
            "LABO-AI-Setup-arm64-helper.sha256",
            "LABO-AI-Setup-x64-helper.exe",
            "LABO-AI-Setup-x64-helper.exe.sha256",
            "LABO-AI-Setup-x64.AppImage",
            "LABO-AI-Setup-x64.AppImage.sha256",
        ];
        Ok(GitHubRelease {
            tarball_url: format!(
                "https://github.com/{REPOSITORY}/archive/refs/tags/{tag_name}.tar.gz"
            ),
            assets: asset_names
                .iter()
                .map(|name| GitHubAsset {
                    name: (*name).to_string(),
                    browser_download_url: format!(
                        "https://github.com/{REPOSITORY}/releases/download/{tag_name}/{name}"
                    ),
                })
                .collect(),
            tag_name,
            revision: None,
        })
    })
}

fn github_revision(reference: &str) -> Result<String, String> {
    let api_result = (|| {
        let commit = client()?
            .get(format!(
                "https://api.github.com/repos/{REPOSITORY}/commits/{reference}"
            ))
            .send()
            .and_then(|response| response.error_for_status())
            .map_err(|error| format!("Unable to resolve GitHub revision {reference}: {error}"))?
            .json::<GitHubCommit>()
            .map_err(|error| format!("Invalid GitHub commit response: {error}"))?;
        valid_github_revision(&commit.sha, reference)
    })();
    api_result.or_else(|api_error| {
        let atom = client()?
            .get(format!(
                "https://github.com/{REPOSITORY}/commits/{reference}.atom"
            ))
            .send()
            .and_then(|response| response.error_for_status())
            .map_err(|fallback_error| {
                format!("{api_error}; Atom fallback failed: {fallback_error}")
            })?
            .text()
            .map_err(|fallback_error| {
                format!("{api_error}; invalid Atom response: {fallback_error}")
            })?;
        atom_revision(&atom, reference)
            .map_err(|fallback_error| format!("{api_error}; {fallback_error}"))
    })
}

fn atom_revision(atom: &str, reference: &str) -> Result<String, String> {
    let marker = "Grit::Commit/";
    let start = atom
        .find(marker)
        .map(|index| index + marker.len())
        .ok_or_else(|| "Atom fallback contained no commit".to_string())?;
    let revision = atom[start..]
        .chars()
        .take_while(|character| character.is_ascii_hexdigit())
        .collect::<String>();
    valid_github_revision(&revision, reference)
}

fn valid_github_revision(revision: &str, reference: &str) -> Result<String, String> {
    if revision.len() < 7
        || !revision
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        return Err(format!(
            "GitHub returned an invalid revision for {reference}"
        ));
    }
    Ok(revision.to_ascii_lowercase())
}

fn main_source() -> Result<GitHubRelease, String> {
    let revision = github_revision("main")?;
    Ok(GitHubRelease {
        tag_name: format!("main@{}", &revision[..7]),
        tarball_url: format!(
            "https://github.com/{REPOSITORY}/archive/{}.tar.gz",
            revision
        ),
        assets: Vec::new(),
        revision: Some(revision),
    })
}

fn normalized_channel(value: Option<&str>) -> &'static str {
    if value == Some("main") {
        "main"
    } else {
        "stable"
    }
}

fn requested_channel() -> &'static str {
    let arguments = env::args().collect::<Vec<_>>();
    let explicit = arguments
        .windows(2)
        .find_map(|pair| (pair[0] == "--channel").then_some(pair[1].as_str()));
    if explicit.is_some() {
        return normalized_channel(explicit);
    }
    normalized_channel(read_state().installed_channel.as_deref())
}

fn install_root() -> Result<PathBuf, String> {
    #[cfg(target_os = "windows")]
    {
        return dirs::data_local_dir()
            .map(|path| path.join("LABO AI").join("setup-data"))
            .ok_or_else(|| "No local application-data directory is available".to_string());
    }
    #[cfg(not(target_os = "windows"))]
    dirs::data_local_dir()
        .map(|path| path.join("LABO AI Setup"))
        .ok_or_else(|| "No local application-data directory is available".to_string())
}

fn electron_user_data() -> Result<PathBuf, String> {
    #[cfg(target_os = "macos")]
    {
        return dirs::home_dir()
            .map(|path| path.join("Library/Application Support/LABO AI"))
            .ok_or_else(|| "No home directory is available".to_string());
    }
    #[cfg(target_os = "windows")]
    {
        return dirs::config_dir()
            .map(|path| path.join("LABO AI"))
            .ok_or_else(|| "No roaming application-data directory is available".to_string());
    }
    #[cfg(target_os = "linux")]
    {
        return dirs::config_dir()
            .map(|path| path.join("LABO AI"))
            .ok_or_else(|| "No XDG configuration directory is available".to_string());
    }
    #[allow(unreachable_code)]
    install_root()
}

fn app_destination() -> Result<PathBuf, String> {
    #[cfg(target_os = "macos")]
    {
        return dirs::home_dir()
            .map(|path| path.join("Applications/LABO AI.app"))
            .ok_or_else(|| "No home directory is available".to_string());
    }
    #[cfg(target_os = "windows")]
    {
        return dirs::data_local_dir()
            .map(|path| path.join("Programs/LABO AI"))
            .ok_or_else(|| "No local application-data directory is available".to_string());
    }
    #[cfg(target_os = "linux")]
    {
        return dirs::data_local_dir()
            .map(|path| path.join("LABO AI").join("app"))
            .ok_or_else(|| "No XDG application-data directory is available".to_string());
    }
    #[allow(unreachable_code)]
    Err("LABO AI Setup currently supports macOS, Windows and Linux".to_string())
}

fn state_path() -> Result<PathBuf, String> {
    Ok(install_root()?.join("install-state.json"))
}

fn read_state() -> InstallState {
    state_path()
        .ok()
        .and_then(|path| fs::read_to_string(path).ok())
        .and_then(|content| serde_json::from_str(&content).ok())
        .unwrap_or_default()
}

fn status(channel: &str) -> Result<SetupStatus, String> {
    let state = read_state();
    let channel = normalized_channel(Some(channel));
    let latest_source = if channel == "main" {
        main_source()
    } else {
        latest_release()
    }
    .ok();
    let latest_revision = latest_source.as_ref().and_then(|release| {
        release
            .revision
            .clone()
            .or_else(|| github_revision(&release.tag_name).ok())
    });
    let latest_tag = latest_source.map(|release| release.tag_name);
    let installed_revision = state.installed_revision.or_else(|| {
        state.installed_tag.as_deref().and_then(|tag| {
            tag.strip_prefix("main@")
                .map(str::to_string)
                .or_else(|| github_revision(tag).ok())
        })
    });
    Ok(SetupStatus {
        installed_tag: state.installed_tag,
        installed_channel: state.installed_channel,
        installed_revision,
        latest_tag,
        latest_revision,
        channel: channel.to_string(),
        app_path: app_destination()?.display().to_string(),
        platform: env::consts::OS,
        setup_version: env!("CARGO_PKG_VERSION"),
        installing: INSTALL_IN_PROGRESS.load(Ordering::Acquire),
    })
}

#[tauri::command]
fn setup_status(channel: Option<String>) -> Result<SetupStatus, String> {
    status(normalized_channel(channel.as_deref()))
}

fn emit(app: &AppHandle, stage: &str, message: impl Into<String>, percent: u8) {
    let _ = app.emit(
        "setup-progress",
        ProgressEvent {
            stage: stage.to_string(),
            message: message.into(),
            percent,
        },
    );
}

fn download(url: &str) -> Result<Vec<u8>, String> {
    let mut response = client()?
        .get(url)
        .send()
        .and_then(|response| response.error_for_status())
        .map_err(|error| format!("Download failed: {error}"))?;
    let mut bytes = Vec::new();
    response
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
}

fn helper_destination() -> Result<PathBuf, String> {
    let extension = if cfg!(target_os = "windows") {
        ".exe"
    } else {
        ""
    };
    Ok(electron_user_data()?
        .join("installer")
        .join(format!("labo-ai-setup{extension}")))
}

fn setup_helper_asset() -> Result<&'static str, String> {
    match (env::consts::OS, env::consts::ARCH) {
        ("macos", "aarch64") => Ok("LABO-AI-Setup-arm64-helper"),
        ("macos", "x86_64") => Ok("LABO-AI-Setup-x64-helper"),
        ("windows", "x86_64") => Ok("LABO-AI-Setup-x64-helper.exe"),
        ("windows", "aarch64") => Ok("LABO-AI-Setup-arm64-helper.exe"),
        ("linux", "x86_64") => Ok("LABO-AI-Setup-x64.AppImage"),
        _ => Err(format!(
            "Unsupported Setup helper platform: {} {}",
            env::consts::OS,
            env::consts::ARCH
        )),
    }
}

fn version_parts(value: &str) -> Option<Vec<u64>> {
    value
        .trim_start_matches('v')
        .split('.')
        .map(str::parse::<u64>)
        .collect::<Result<Vec<_>, _>>()
        .ok()
}

fn release_is_newer(tag: &str) -> bool {
    match (version_parts(tag), version_parts(env!("CARGO_PKG_VERSION"))) {
        (Some(latest), Some(current)) => latest > current,
        _ => false,
    }
}

fn relaunch_latest_setup(
    app: &AppHandle,
    release: &GitHubRelease,
    channel: &str,
) -> Result<bool, String> {
    if !release_is_newer(&release.tag_name) {
        return Ok(false);
    }
    let asset_name = setup_helper_asset()?;
    let checksum_name = format!("{asset_name}.sha256");
    let helper_asset = release
        .assets
        .iter()
        .find(|asset| asset.name == asset_name)
        .ok_or_else(|| format!("Latest release does not contain {asset_name}"))?;
    let checksum_asset = release
        .assets
        .iter()
        .find(|asset| asset.name == checksum_name)
        .ok_or_else(|| format!("Latest release does not contain {checksum_name}"))?;

    emit(
        app,
        "Setup update",
        format!(
            "Updating LABO AI Setup to {} before continuing…",
            release.tag_name
        ),
        8,
    );
    let expected = String::from_utf8(download(&checksum_asset.browser_download_url)?)
        .map_err(|error| error.to_string())?
        .split_whitespace()
        .next()
        .ok_or_else(|| "Setup checksum file is empty".to_string())?
        .to_lowercase();
    let bytes = download(&helper_asset.browser_download_url)?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected {
        return Err(format!("SHA-256 mismatch for {asset_name}"));
    }

    let destination = helper_destination()?;
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let next = destination.with_file_name(if cfg!(target_os = "windows") {
        "labo-ai-setup-next.exe"
    } else {
        "labo-ai-setup-next"
    });
    fs::write(&next, bytes).map_err(|error| format!("Unable to stage the new Setup: {error}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&next, fs::Permissions::from_mode(0o755))
            .map_err(|error| error.to_string())?;
    }
    let mut command = background_command(next);
    #[cfg(target_os = "linux")]
    command.env("APPIMAGE_EXTRACT_AND_RUN", "1");
    command
        .arg("--auto-install")
        .arg("--channel")
        .arg(channel)
        .spawn()
        .map_err(|error| format!("Unable to relaunch the updated Setup: {error}"))?;
    Ok(true)
}

fn only_child_directory(path: &Path) -> Result<PathBuf, String> {
    fs::read_dir(path)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|entry| entry.is_dir())
        .ok_or_else(|| format!("Archive did not contain a directory: {}", path.display()))
}

fn extract_tar_gz(bytes: &[u8], destination: &Path) -> Result<PathBuf, String> {
    fs::create_dir_all(destination).map_err(|error| error.to_string())?;
    let decoder = GzDecoder::new(Cursor::new(bytes));
    let mut archive = tar::Archive::new(decoder);
    archive
        .unpack(destination)
        .map_err(|error| format!("Archive extraction failed: {error}"))?;
    only_child_directory(destination)
}

fn extract_zip(bytes: &[u8], destination: &Path) -> Result<PathBuf, String> {
    fs::create_dir_all(destination).map_err(|error| error.to_string())?;
    let mut archive =
        zip::ZipArchive::new(Cursor::new(bytes)).map_err(|error| error.to_string())?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|error| error.to_string())?;
        let Some(relative) = entry.enclosed_name() else {
            continue;
        };
        let output = destination.join(relative);
        if entry.is_dir() {
            fs::create_dir_all(&output).map_err(|error| error.to_string())?;
        } else {
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
            let mut file = fs::File::create(output).map_err(|error| error.to_string())?;
            io::copy(&mut entry, &mut file).map_err(|error| error.to_string())?;
        }
    }
    only_child_directory(destination)
}

fn node_asset() -> Result<(&'static str, &'static str), String> {
    match (env::consts::OS, env::consts::ARCH) {
        ("macos", "aarch64") => Ok(("node-v22.14.0-darwin-arm64.tar.gz", "tar.gz")),
        ("macos", "x86_64") => Ok(("node-v22.14.0-darwin-x64.tar.gz", "tar.gz")),
        ("windows", "x86_64") => Ok(("node-v22.14.0-win-x64.zip", "zip")),
        ("windows", "aarch64") => Ok(("node-v22.14.0-win-arm64.zip", "zip")),
        ("linux", "x86_64") => Ok(("node-v22.14.0-linux-x64.tar.gz", "tar.gz")),
        ("linux", "aarch64") => Ok(("node-v22.14.0-linux-arm64.tar.gz", "tar.gz")),
        _ => Err(format!(
            "Unsupported platform: {} {}",
            env::consts::OS,
            env::consts::ARCH
        )),
    }
}

fn ensure_node(app: &AppHandle) -> Result<PathBuf, String> {
    let runtime = install_root()?.join(format!(
        "runtime/node-{NODE_VERSION}-{}-{}",
        env::consts::OS,
        env::consts::ARCH
    ));
    let npm = if cfg!(target_os = "windows") {
        runtime.join("npm.cmd")
    } else {
        runtime.join("bin/npm")
    };
    if npm.exists() {
        return Ok(runtime);
    }

    emit(app, "Runtime", "Downloading the managed Node.js build…", 22);
    let (asset, kind) = node_asset()?;
    let base = format!("https://nodejs.org/dist/{NODE_VERSION}");
    let checksums = String::from_utf8(download(&format!("{base}/SHASUMS256.txt"))?)
        .map_err(|error| error.to_string())?;
    let expected = checksums
        .lines()
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            let digest = parts.next()?;
            let filename = parts.next()?.trim_start_matches('*');
            (filename == asset).then(|| digest.to_string())
        })
        .ok_or_else(|| format!("No official Node.js checksum found for {asset}"))?;
    let bytes = download(&format!("{base}/{asset}"))?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected {
        return Err(format!("Node.js checksum mismatch for {asset}"));
    }

    let temporary = TempDir::new_in(install_root()?).map_err(|error| error.to_string())?;
    let extracted = if kind == "zip" {
        extract_zip(&bytes, temporary.path())?
    } else {
        extract_tar_gz(&bytes, temporary.path())?
    };
    if let Some(parent) = runtime.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::rename(extracted, &runtime)
        .map_err(|error| format!("Cannot install Node.js runtime: {error}"))?;
    Ok(runtime)
}

fn run_npm(node_root: &Path, source: &Path, arguments: &[&str]) -> Result<(), String> {
    let npm = if cfg!(target_os = "windows") {
        node_root.join("npm.cmd")
    } else {
        node_root.join("bin/npm")
    };
    let node_bin = if cfg!(target_os = "windows") {
        node_root.to_path_buf()
    } else {
        node_root.join("bin")
    };
    let mut search_paths = vec![node_bin];
    if let Some(current) = env::var_os("PATH") {
        search_paths.extend(env::split_paths(&current));
    }
    let output = background_command(&npm)
        .args(arguments)
        .current_dir(source)
        .env(
            "PATH",
            env::join_paths(search_paths).map_err(|error| error.to_string())?,
        )
        .output()
        .map_err(|error| format!("Unable to run npm: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let tail = stderr
        .lines()
        .rev()
        .take(18)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");
    Err(format!(
        "npm command failed ({})\n{tail}",
        arguments.join(" ")
    ))
}

#[cfg(target_os = "macos")]
fn built_application(source: &Path) -> PathBuf {
    let folder = if env::consts::ARCH == "aarch64" {
        "mac-arm64"
    } else {
        "mac"
    };
    source.join("release").join(folder).join("LABO AI.app")
}

#[cfg(target_os = "windows")]
fn built_application(source: &Path) -> PathBuf {
    let folder = if env::consts::ARCH == "aarch64" {
        "win-arm64-unpacked"
    } else {
        "win-unpacked"
    };
    source.join("release").join(folder)
}

#[cfg(target_os = "linux")]
fn built_application(source: &Path) -> PathBuf {
    let folder = if env::consts::ARCH == "aarch64" {
        "linux-arm64-unpacked"
    } else {
        "linux-unpacked"
    };
    source.join("release").join(folder)
}

fn copy_application(source: &Path, destination: &Path) -> Result<(), String> {
    if destination.exists() {
        fs::remove_dir_all(destination).map_err(|error| error.to_string())?;
    }
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    let status = background_command("/bin/cp")
        .arg("-R")
        .arg(source)
        .arg(destination)
        .status();
    #[cfg(target_os = "windows")]
    let status = background_command("robocopy")
        .arg(source)
        .arg(destination)
        .arg("/E")
        .arg("/NFL")
        .arg("/NDL")
        .status();
    let status = status.map_err(|error| format!("Unable to copy LABO AI: {error}"))?;
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    let success = status.success();
    #[cfg(target_os = "windows")]
    let success = status.code().is_some_and(|code| code <= 7);
    if success {
        Ok(())
    } else {
        Err(format!("Application copy failed with {status}"))
    }
}

fn install_helper() -> Result<(), String> {
    let destination = helper_destination()?;
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let current_executable = env::current_exe().map_err(|error| error.to_string())?;
    #[cfg(target_os = "linux")]
    let current = env::var_os("APPIMAGE")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .unwrap_or(current_executable);
    #[cfg(not(target_os = "linux"))]
    let current = current_executable;
    if current != destination {
        fs::copy(current, &destination)
            .map_err(|error| format!("Unable to install update helper: {error}"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&destination, fs::Permissions::from_mode(0o755))
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn launch_application(destination: &Path) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    background_command("/usr/bin/open")
        .arg(destination)
        .spawn()
        .map_err(|error| error.to_string())?;
    #[cfg(target_os = "windows")]
    background_command(destination.join("LABO AI.exe"))
        .spawn()
        .map_err(|error| error.to_string())?;
    #[cfg(target_os = "linux")]
    background_command(linux_application_executable(destination)?)
        .spawn()
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn linux_application_executable(destination: &Path) -> Result<PathBuf, String> {
    ["labo-ai", "LABO AI"]
        .iter()
        .map(|name| destination.join(name))
        .find(|path| path.is_file())
        .ok_or_else(|| {
            format!(
                "The installed Linux executable is missing in {}",
                destination.display()
            )
        })
}

#[cfg(target_os = "linux")]
fn install_linux_integration(destination: &Path, source: &Path) -> Result<(), String> {
    let executable = linux_application_executable(destination)?;
    let icon = destination.join("resources").join("labo-ai.png");
    if let Some(parent) = icon.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::copy(source.join("build/icon.png"), &icon)
        .map_err(|error| format!("Unable to install the Linux application icon: {error}"))?;

    let applications = dirs::data_local_dir()
        .map(|path| path.join("applications"))
        .ok_or_else(|| "No XDG application-data directory is available".to_string())?;
    fs::create_dir_all(&applications).map_err(|error| error.to_string())?;
    let desktop_entry = format!(
        "[Desktop Entry]\nType=Application\nVersion=1.0\nName=LABO AI\nComment=Atomic neural architecture laboratory\nExec=\"{}\"\nIcon={}\nTerminal=false\nCategories=Development;Science;\nStartupWMClass=LABO AI\n",
        executable.display().to_string().replace('"', "\\\""),
        icon.display()
    );
    fs::write(applications.join("labo-ai.desktop"), desktop_entry)
        .map_err(|error| format!("Unable to create the Linux application launcher: {error}"))?;

    if background_command("update-desktop-database")
        .arg(&applications)
        .status()
        .is_err()
    {
        // The desktop entry is still discovered without the optional cache refresh utility.
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn powershell_literal(value: &Path) -> String {
    value.display().to_string().replace('\'', "''")
}

#[cfg(target_os = "windows")]
fn run_powershell(script: &str) -> Result<(), String> {
    let status = background_command("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ])
        .status()
        .map_err(|error| format!("Unable to configure the Windows application entry: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "Windows application integration failed with status {status}"
        ))
    }
}

#[cfg(target_os = "windows")]
fn install_windows_integration(destination: &Path, release_tag: &str) -> Result<(), String> {
    let executable = destination.join("LABO AI.exe");
    if !executable.is_file() {
        return Err(format!(
            "The installed Windows executable is missing at {}",
            executable.display()
        ));
    }

    let uninstall_script = install_root()?.join("uninstall-labo-ai.ps1");
    let uninstall_body = format!(
        "$ErrorActionPreference = 'SilentlyContinue'\n\
         Stop-Process -Name 'LABO AI' -Force\n\
         Start-Sleep -Milliseconds 400\n\
         Remove-Item -LiteralPath '{}' -Recurse -Force\n\
         $shell = New-Object -ComObject WScript.Shell\n\
         Remove-Item -LiteralPath ([IO.Path]::Combine([Environment]::GetFolderPath('Programs'), 'LABO AI.lnk')) -Force\n\
         Remove-Item -LiteralPath ([IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'LABO AI.lnk')) -Force\n\
         Remove-Item -LiteralPath 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\LABO AI' -Recurse -Force\n",
        powershell_literal(destination)
    );
    fs::write(&uninstall_script, uninstall_body)
        .map_err(|error| format!("Unable to create the LABO AI uninstaller: {error}"))?;

    let executable_literal = powershell_literal(&executable);
    let destination_literal = powershell_literal(destination);
    let uninstall_literal = powershell_literal(&uninstall_script);
    let display_version = release_tag.trim_start_matches('v').replace('\'', "''");
    let script = format!(
        "$ErrorActionPreference = 'Stop'; \
         $exe = '{executable_literal}'; \
         $shell = New-Object -ComObject WScript.Shell; \
         foreach ($link in @([IO.Path]::Combine([Environment]::GetFolderPath('Programs'), 'LABO AI.lnk'), [IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'LABO AI.lnk'))) {{ \
           $shortcut = $shell.CreateShortcut($link); \
           $shortcut.TargetPath = $exe; \
           $shortcut.WorkingDirectory = '{destination_literal}'; \
           $shortcut.IconLocation = \"$exe,0\"; \
           $shortcut.Description = 'LABO AI neural architecture laboratory'; \
           $shortcut.Save(); \
         }}; \
         $key = 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\LABO AI'; \
         New-Item -Path $key -Force | Out-Null; \
         New-ItemProperty -Path $key -Name DisplayName -Value 'LABO AI' -PropertyType String -Force | Out-Null; \
         New-ItemProperty -Path $key -Name DisplayVersion -Value '{display_version}' -PropertyType String -Force | Out-Null; \
         New-ItemProperty -Path $key -Name Publisher -Value 'Complexity-ML' -PropertyType String -Force | Out-Null; \
         New-ItemProperty -Path $key -Name InstallLocation -Value '{destination_literal}' -PropertyType String -Force | Out-Null; \
         New-ItemProperty -Path $key -Name DisplayIcon -Value \"$exe,0\" -PropertyType String -Force | Out-Null; \
         New-ItemProperty -Path $key -Name URLInfoAbout -Value 'https://www.complexity-ai.fr/labo-ai' -PropertyType String -Force | Out-Null; \
         New-ItemProperty -Path $key -Name UninstallString -Value 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{uninstall_literal}\"' -PropertyType String -Force | Out-Null; \
         New-ItemProperty -Path $key -Name NoModify -Value 1 -PropertyType DWord -Force | Out-Null; \
         New-ItemProperty -Path $key -Name NoRepair -Value 1 -PropertyType DWord -Force | Out-Null"
    );
    run_powershell(&script)
}

#[cfg(target_os = "windows")]
fn schedule_windows_setup_cleanup() -> Result<(), String> {
    let process_id = std::process::id();
    let script = format!(
        "$ErrorActionPreference = 'SilentlyContinue'; \
         Wait-Process -Id {process_id}; \
         Start-Sleep -Milliseconds 500; \
         Remove-Item -LiteralPath ([IO.Path]::Combine([Environment]::GetFolderPath('Programs'), 'LABO AI Setup.lnk')) -Force; \
         Remove-Item -LiteralPath ([IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'LABO AI Setup.lnk')) -Force; \
         $setupDirectories = @(); \
         Get-ItemProperty 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*' | \
           Where-Object {{ $_.DisplayName -eq 'LABO AI Setup' }} | \
           ForEach-Object {{ \
             if ($_.InstallLocation -and ([IO.Path]::GetFileName($_.InstallLocation.TrimEnd('\\')) -eq 'LABO AI Setup')) {{ $setupDirectories += $_.InstallLocation }}; \
             Remove-Item -LiteralPath $_.PSPath -Recurse -Force \
           }}; \
         foreach ($directory in $setupDirectories) {{ \
           if (Test-Path -LiteralPath $directory) {{ Remove-Item -LiteralPath $directory -Recurse -Force }} \
         }}"
    );
    background_command("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            &script,
        ])
        .spawn()
        .map_err(|error| format!("Unable to schedule Setup cleanup: {error}"))?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn prepare_application_handoff(app: &AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("main") {
        window
            .hide()
            .map_err(|error| format!("Unable to hide LABO AI Setup: {error}"))?;
    }
    app.set_activation_policy(tauri::ActivationPolicy::Accessory)
        .map_err(|error| format!("Unable to remove LABO AI Setup from the Dock: {error}"))?;
    thread::sleep(Duration::from_millis(250));
    Ok(())
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn prepare_application_handoff(_app: &AppHandle) -> Result<(), String> {
    if let Some(window) = _app.get_webview_window("main") {
        window
            .hide()
            .map_err(|error| format!("Unable to hide LABO AI Setup: {error}"))?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn restore_setup_after_handoff_failure(app: &AppHandle) {
    let _ = app.set_activation_policy(tauri::ActivationPolicy::Regular);
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn restore_setup_after_handoff_failure(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

#[cfg(target_os = "windows")]
fn stop_running_application() -> Result<(), String> {
    let destination = app_destination()?;
    let destination_literal = powershell_literal(&destination);
    let script = format!(
        "$root = '{destination_literal}'.TrimEnd('\\'); \
         Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | \
           Where-Object {{ $_.ExecutablePath -and ($_.ExecutablePath.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) }} | \
           ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}"
    );
    let _ = background_command("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            &script,
        ])
        .output();
    let _ = background_command("taskkill")
        .args(["/IM", "LABO AI.exe", "/T", "/F"])
        .output();
    for _ in 0..50 {
        let running = background_command("tasklist")
            .args(["/FI", "IMAGENAME eq LABO AI.exe", "/NH"])
            .output()
            .map(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .to_ascii_lowercase()
                    .contains("labo ai.exe")
            })
            .unwrap_or(false);
        if !running {
            thread::sleep(Duration::from_millis(350));
            return Ok(());
        }
        thread::sleep(Duration::from_millis(200));
    }
    Err("LABO AI is still running. Close the application, then retry the update.".to_string())
}

#[cfg(not(target_os = "windows"))]
fn stop_running_application() -> Result<(), String> {
    Ok(())
}

fn retryable_filesystem_error(error: &io::Error) -> bool {
    #[cfg(target_os = "windows")]
    {
        return matches!(error.raw_os_error(), Some(5 | 32 | 33))
            || error.kind() == io::ErrorKind::PermissionDenied;
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = error;
        false
    }
}

fn retry_filesystem_operation(mut operation: impl FnMut() -> io::Result<()>) -> io::Result<()> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match operation() {
            Ok(()) => return Ok(()),
            Err(error) if retryable_filesystem_error(&error) && Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(250));
            }
            Err(error) => return Err(error),
        }
    }
}

fn remove_application_directory(path: &Path) -> io::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    retry_filesystem_operation(|| fs::remove_dir_all(path))
}

fn rename_application_directory(source: &Path, destination: &Path) -> io::Result<()> {
    retry_filesystem_operation(|| fs::rename(source, destination))
}

fn activate_application(next: &Path, destination: &Path, previous: &Path) -> Result<(), String> {
    if previous.exists() {
        remove_application_directory(previous)
            .map_err(|error| format!("Unable to remove the previous rollback copy: {error}"))?;
    }
    let had_previous = destination.exists();
    if had_previous {
        rename_application_directory(destination, previous)
            .map_err(|error| format!("Unable to preserve the previous LABO AI build: {error}"))?;
    }
    if let Err(error) = rename_application_directory(next, destination) {
        if had_previous && previous.exists() {
            let _ = rename_application_directory(previous, destination);
        }
        return Err(format!(
            "Unable to activate the new LABO AI build; the previous build was restored: {error}"
        ));
    }
    Ok(())
}

fn perform_install(app: &AppHandle, channel: &str) -> Result<InstallResult, String> {
    fs::create_dir_all(install_root()?).map_err(|error| error.to_string())?;
    let channel = normalized_channel(Some(channel));
    emit(
        app,
        "Release",
        if channel == "main" {
            "Checking the latest LABO AI main commit…"
        } else {
            "Checking the latest LABO AI release…"
        },
        5,
    );
    let helper_release = latest_release()?;
    if relaunch_latest_setup(app, &helper_release, channel)? {
        return Ok(InstallResult {
            tag: helper_release.tag_name,
            path: String::new(),
            setup_relaunched: true,
        });
    }
    let release = if channel == "main" {
        main_source()?
    } else {
        helper_release
    };

    emit(
        app,
        "Source",
        format!("Downloading source for {}…", release.tag_name),
        12,
    );
    let source_bytes = download(&release.tarball_url)?;
    let source_temp = TempDir::new_in(install_root()?).map_err(|error| error.to_string())?;
    let source = extract_tar_gz(&source_bytes, source_temp.path())?;
    let node = ensure_node(app)?;

    emit(
        app,
        "Dependencies",
        "Installing locked JavaScript dependencies…",
        36,
    );
    run_npm(&node, &source, &["ci"])?;
    emit(app, "Runtime", "Preparing the isolated Python runtime…", 45);
    let python_runtime = electron_user_data()?.join("runtime");
    let python_runtime_argument = python_runtime.to_string_lossy().into_owned();
    run_npm(
        &node,
        &source,
        &[
            "run",
            "runtime:setup",
            "--",
            "--venv",
            &python_runtime_argument,
        ],
    )?;
    emit(
        app,
        "Build",
        "Building the LABO AI interface and secure desktop bridge…",
        55,
    );
    #[cfg(target_os = "macos")]
    run_npm(
        &node,
        &source,
        &[
            "run",
            "package:mac:dir",
            "--",
            if env::consts::ARCH == "aarch64" {
                "--arm64"
            } else {
                "--x64"
            },
        ],
    )?;
    #[cfg(target_os = "windows")]
    run_npm(
        &node,
        &source,
        &[
            "run",
            "package:win:dir",
            "--",
            if env::consts::ARCH == "aarch64" {
                "--arm64"
            } else {
                "--x64"
            },
        ],
    )?;
    #[cfg(target_os = "linux")]
    run_npm(
        &node,
        &source,
        &[
            "run",
            "package:linux:dir",
            "--",
            if env::consts::ARCH == "aarch64" {
                "--arm64"
            } else {
                "--x64"
            },
        ],
    )?;

    let built = built_application(&source);
    if !built.exists() {
        return Err(format!(
            "The desktop build was not produced at {}",
            built.display()
        ));
    }
    let destination = app_destination()?;
    let next = destination.with_extension("next");
    let previous = destination.with_extension("previous");
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    emit(
        app,
        "Install",
        "Installing the new application and keeping one rollback copy…",
        88,
    );
    copy_application(&built, &next)?;
    stop_running_application()?;
    activate_application(&next, &destination, &previous)?;

    #[cfg(target_os = "windows")]
    install_windows_integration(&destination, &release.tag_name)?;
    #[cfg(target_os = "linux")]
    install_linux_integration(&destination, &source)?;

    #[cfg(target_os = "macos")]
    {
        let signed = background_command("/usr/bin/codesign")
            .args(["--force", "--deep", "--sign", "-"])
            .arg(&destination)
            .status();
        if !signed.is_ok_and(|status| status.success()) {
            return Err(
                "The locally installed macOS application could not be ad-hoc signed".to_string(),
            );
        }
    }

    install_helper()?;
    let installed_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let state = InstallState {
        installed_tag: Some(release.tag_name.clone()),
        installed_channel: Some(channel.to_string()),
        installed_revision: release
            .revision
            .clone()
            .or_else(|| github_revision(&release.tag_name).ok()),
        installed_at: Some(installed_at),
        app_path: Some(destination.display().to_string()),
    };
    fs::write(
        state_path()?,
        serde_json::to_vec_pretty(&state).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    emit(
        app,
        "Complete",
        "LABO AI is ready. Launching the application…",
        100,
    );
    prepare_application_handoff(app)?;
    if let Err(error) = launch_application(&destination) {
        restore_setup_after_handoff_failure(app);
        return Err(error);
    }
    #[cfg(target_os = "windows")]
    schedule_windows_setup_cleanup()?;
    Ok(InstallResult {
        tag: release.tag_name,
        path: destination.display().to_string(),
        setup_relaunched: false,
    })
}

#[tauri::command]
async fn install_latest(app: AppHandle, channel: Option<String>) -> Result<InstallResult, String> {
    let worker_app = app.clone();
    let channel = normalized_channel(channel.as_deref()).to_string();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let _guard = begin_install()?;
        perform_install(&worker_app, &channel)
    })
    .await
    .map_err(|error| error.to_string())??;
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.close();
    }
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        app.exit(0);
    });
    Ok(result)
}

pub fn run() {
    if env::args().any(|argument| argument == "--status") {
        match status(requested_channel())
            .and_then(|value| serde_json::to_string(&value).map_err(|error| error.to_string()))
        {
            Ok(value) => println!("{value}"),
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
        return;
    }

    let automatic_install = env::args().any(|argument| argument == "--auto-install");
    let automatic_channel = requested_channel().to_string();
    tauri::Builder::default()
        .setup(move |app| {
            #[cfg(target_os = "macos")]
            app.handle()
                .set_activation_policy(tauri::ActivationPolicy::Accessory)?;

            if let Some(window) = app.get_webview_window("main") {
                window.show()?;
                window.set_focus()?;
            }

            if automatic_install {
                let handle = app.handle().clone();
                match begin_install() {
                    Ok(install_guard) => {
                        thread::spawn(move || {
                            let _guard = install_guard;
                            thread::sleep(Duration::from_millis(700));
                            if let Err(error) = perform_install(&handle, &automatic_channel) {
                                emit(&handle, "Failed", error, 100);
                                return;
                            }
                            if let Some(window) = handle.get_webview_window("main") {
                                let _ = window.close();
                            }
                            thread::sleep(Duration::from_millis(50));
                            handle.exit(0);
                        });
                    }
                    Err(error) => emit(&handle, "Failed", error, 100),
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![setup_status, install_latest])
        .run(tauri::generate_context!())
        .expect("error while running LABO AI Setup");
}

#[cfg(test)]
mod tests {
    use super::{activate_application, atom_revision, release_is_newer, version_parts};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn reads_the_head_commit_from_the_public_github_atom_feed() {
        let atom = "<entry><id>tag:github.com,2008:Grit::Commit/03fb96e745f18e7c30886dbee66db21ad682be08</id></entry>";
        assert_eq!(
            atom_revision(atom, "main").unwrap(),
            "03fb96e745f18e7c30886dbee66db21ad682be08"
        );
    }

    #[test]
    fn compares_setup_versions_without_downgrading() {
        assert_eq!(version_parts("v1.4.12"), Some(vec![1, 4, 12]));
        assert!(release_is_newer("v999.0.0"));
        assert!(!release_is_newer(env!("CARGO_PKG_VERSION")));
        assert!(!release_is_newer("v0.0.1"));
    }

    #[test]
    fn activates_a_build_and_keeps_one_rollback_copy() {
        let root = tempdir().unwrap();
        let destination = root.path().join("LABO AI");
        let next = root.path().join("LABO AI.next");
        let previous = root.path().join("LABO AI.previous");
        fs::create_dir_all(&destination).unwrap();
        fs::create_dir_all(&next).unwrap();
        fs::write(destination.join("version.txt"), "old").unwrap();
        fs::write(next.join("version.txt"), "new").unwrap();

        activate_application(&next, &destination, &previous).unwrap();

        assert_eq!(
            fs::read_to_string(destination.join("version.txt")).unwrap(),
            "new"
        );
        assert_eq!(
            fs::read_to_string(previous.join("version.txt")).unwrap(),
            "old"
        );
        assert!(!next.exists());
    }
}
