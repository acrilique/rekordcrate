// Copyright (c) 2026 Jan Holthuis <jan.holthuis@rub.de>
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at
// http://mozilla.org/MPL/2.0/.
//
// SPDX-License-Identifier: MPL-2.0

use binrw::BinRead;
use clap::{Parser, Subcommand};
use fallible_iterator::FallibleIterator;
use lofty::prelude::*;
use rekordcrate::device::get_playlists;
use rekordcrate::pdb::io::Database;
use rekordcrate::pdb::*;
use rekordcrate::setting::{Setting, SettingType};
use rekordcrate::xml::Document;
use rekordcrate::{anlz::ANLZ, util::TableIndex};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(author, version, about)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List the playlist tree from a Pioneer Database (`.PDB`) file.
    ListPlaylists {
        /// File to parse.
        #[arg(value_name = "PDB_FILE")]
        path: PathBuf,
    },
    /// Display settings from a Rekordbox device export.
    ListSettings {
        /// Path to parse.
        #[arg(value_name = "EXPORT_PATH")]
        path: PathBuf,
    },
    /// Export the playlists from a Pioneer Database (`.PDB`) file to M3U files.
    ExportPlaylists {
        /// File to parse.
        #[arg(value_name = "EXPORT_PATH")]
        path: PathBuf,
        /// Output directory to write M3U files to.
        #[arg(value_name = "OUTPUT_DIR")]
        output_dir: PathBuf,
    },
    /// Parse and dump a Rekordbox Analysis (`ANLZXXXX.DAT`) file.
    DumpANLZ {
        /// File to parse.
        #[arg(value_name = "ANLZ_FILE")]
        path: PathBuf,
    },
    /// Parse and dump a Pioneer Database (`.PDB`) file.
    DumpPDB {
        /// File to parse.
        #[arg(value_name = "PDB_FILE")]
        path: PathBuf,
        /// Database type: "plain" (export.pdb) or "ext" (exportExt.pdb). Tries to guess based on file name of not specified.
        #[arg(long, value_name = "DB_TYPE", value_parser = ["plain", "ext"])]
        db_type: Option<String>,
        /// Attempt to parse unknown table types instead of skipping them.
        #[arg(long)]
        parse_unknown_tables: bool,
    },
    /// Parse and dump a Pioneer Settings (`*SETTING.DAT`) file.
    DumpSetting {
        /// File to parse.
        #[arg(value_name = "SETTING_FILE")]
        path: PathBuf,
        /// Setting type.
        #[arg(long, value_name = "SETTING_TYPE", value_parser = ["devsetting", "djmmysetting", "mysetting", "mysetting2"])]
        setting_type: Option<String>,
    },
    /// Parse and dump a Pioneer XML (`*.xml`) file.
    DumpXML {
        /// File to parse.
        #[arg(value_name = "XML_FILE")]
        path: PathBuf,
    },
    /// Export MP3 files to a Pioneer-compatible USB structure with a PDB database.
    Export {
        /// Input directory containing MP3 files.
        #[arg(value_name = "INPUT_DIR")]
        input_dir: PathBuf,
        /// Output directory for the Pioneer export (default: current directory).
        #[arg(value_name = "OUTPUT_DIR", default_value = ".")]
        output_dir: PathBuf,
    },
}

fn list_playlists(path: &Path) -> rekordcrate::Result<()> {
    use rekordcrate::pdb::{PlaylistTreeNode, PlaylistTreeNodeId};
    use std::collections::{BTreeMap, HashMap};

    let reader = File::open(path)?;
    let mut db = Database::open_non_persistent(reader, DatabaseType::Plain)?;

    let mut playlist_tree: HashMap<PlaylistTreeNodeId, Vec<PlaylistTreeNode>> = HashMap::new();
    let mut playlist_entries: HashMap<PlaylistTreeNodeId, BTreeMap<u32, TrackId>> = HashMap::new();
    let mut artists: HashMap<ArtistId, Artist> = HashMap::new();
    let mut tracks: HashMap<TrackId, Track> = HashMap::new();

    db.iter_rows::<PlaylistTreeNode>()?.for_each(|tree_node| {
        playlist_tree
            .entry(tree_node.parent_id)
            .or_default()
            .push(tree_node.clone());
        Ok(())
    })?;

    db.iter_rows::<Artist>()?.for_each(|artist| {
        artists.insert(artist.id, artist.clone());
        Ok(())
    })?;

    db.iter_rows::<Track>()?.for_each(|track| {
        tracks.insert(track.id, track.clone());
        Ok(())
    })?;

    db.iter_rows::<PlaylistEntry>()?.for_each(|entry| {
        playlist_entries
            .entry(entry.playlist_id)
            .or_default()
            .insert(entry.entry_index, entry.track_id);
        Ok(())
    })?;

    fn print_track(
        track_id: &TrackId,
        artists: &HashMap<ArtistId, Artist>,
        tracks: &HashMap<TrackId, Track>,
    ) {
        let track = match tracks.get(track_id) {
            Some(track) => track,
            None => {
                println!("<Track for {track_id:?} not found>");
                return;
            }
        };
        let artist = match artists.get(&track.artist_id) {
            Some(artist) => artist,
            None => {
                println!(
                    "<Artist for {:?} not found> - {}",
                    &track.artist_id, track.offsets.title
                );
                return;
            }
        };
        println!("{} - {}", artist.offsets.name, track.offsets.title)
    }
    fn print_children_of(
        tree: &HashMap<PlaylistTreeNodeId, Vec<PlaylistTreeNode>>,
        tree_entries: &HashMap<PlaylistTreeNodeId, BTreeMap<u32, TrackId>>,
        artists: &HashMap<ArtistId, Artist>,
        tracks: &HashMap<TrackId, Track>,
        id: PlaylistTreeNodeId,
        level: usize,
    ) {
        tree.get(&id)
            .iter()
            .flat_map(|nodes| nodes.iter())
            .for_each(|node| {
                let indentation = "    ".repeat(level);
                println!(
                    "{}{} {}",
                    indentation,
                    if node.is_folder() { "🗀" } else { "🗎" },
                    node.name,
                );
                if let Some(playlist_tracks) = tree_entries.get(&node.id) {
                    for (index, track_id) in playlist_tracks.iter() {
                        print!("{}  ♫ {}: ", indentation, index);
                        print_track(track_id, artists, tracks);
                    }
                }
                print_children_of(tree, tree_entries, artists, tracks, node.id, level + 1);
            });
    }

    print_children_of(
        &playlist_tree,
        &playlist_entries,
        &artists,
        &tracks,
        PlaylistTreeNodeId(0),
        0,
    );

    Ok(())
}

fn export_playlists(path: &Path, output_dir: &Path) -> rekordcrate::Result<()> {
    use rekordcrate::device::{Playlist, PlaylistNode};
    use rekordcrate::DeviceExportLoader;
    use std::collections::HashMap;
    use std::io::Write;

    let loader = DeviceExportLoader::new(path.into());
    let export_path = loader.get_path();
    let mut db = loader.open_pdb_non_persistent()?;

    let playlists = get_playlists(&mut db)?;
    let playlist_entries = db
        .iter_rows::<PlaylistEntry>()?
        .map(|entry| Ok((entry.playlist_id, (entry.entry_index, entry.track_id))))
        .fold(
            HashMap::<PlaylistTreeNodeId, BTreeMap<u32, TrackId>>::new(),
            |mut acc, (playlist_id, entry)| {
                // BTreeMap keeps entries sorted by their index in the playlist.
                acc.entry(playlist_id).or_default().insert(entry.0, entry.1);
                Ok(acc)
            },
        )?;

    let tracks = db
        .iter_rows::<Track>()?
        .map(|track| Ok((track.id, track.offsets.file_path.clone().into_string()?)))
        .collect::<HashMap<_, _>>()?;

    fn walk_tree(
        node: PlaylistNode,
        path: &Path,
        visit: &impl Fn(Playlist, &Path) -> rekordcrate::Result<()>,
    ) -> rekordcrate::Result<()> {
        match node {
            PlaylistNode::Folder(folder) => folder
                .children
                .into_iter()
                .try_for_each(|child| walk_tree(child, &path.join(&folder.name), visit)),
            PlaylistNode::Playlist(playlist) => visit(playlist, path),
        }
    }

    playlists.into_iter().try_for_each(|node| {
        walk_tree(node, output_dir, &|playlist, path| {
            std::fs::create_dir_all(path)?;
            let playlist_path = path.join(format!("{}.m3u", playlist.name));

            println!("{}", playlist_path.display());
            let mut file = File::create(playlist_path)?;
            if let Some(entries) = playlist_entries.get(&playlist.id) {
                for track in entries.values().filter_map(|track_id| tracks.get(track_id)) {
                    writeln!(
                        &mut file,
                        "{}",
                        export_path
                            .canonicalize()?
                            .join(track.strip_prefix('/').unwrap_or(track))
                            .display(),
                    )?;
                }
            }
            Ok(())
        })
    })?;

    Ok(())
}

fn list_settings(path: &Path) -> rekordcrate::Result<()> {
    use rekordcrate::DeviceExportLoader;

    let loader = DeviceExportLoader::new(path.into());
    let settings = loader.load_settings();

    print!("{}", settings);

    Ok(())
}

fn dump_anlz(path: &Path) -> rekordcrate::Result<()> {
    let mut reader = File::open(path)?;
    let anlz = ANLZ::read(&mut reader)?;
    println!("{:#?}", anlz);

    Ok(())
}

fn dump_pdb(path: &Path, typ: DatabaseType, parse_unknown_tables: bool) -> rekordcrate::Result<()> {
    let reader = File::open(path)?;
    let mut db = Database::open_non_persistent(reader, typ)?;

    println!("{:#?}", db.get_header());

    fn dump_table(db: &mut Database<File>, id: TableIndex) -> rekordcrate::Result<()> {
        let mut page_iter = db.iter_pages_for_table(id)?;
        while let Some(page) = page_iter.next()? {
            println!("  {:?}", page);
            match &page.content {
                PageContent::Data(data_content) => {
                    for row in data_content.rows.values() {
                        println!("      {:?}", row);
                    }
                }
                PageContent::Index(index_content) => {
                    println!("    {:?}", index_content);
                    for entry in index_content.entries.iter() {
                        println!("      {:?}", entry);
                    }
                }
            }
        }
        Ok(())
    }

    let tables = db.get_header().tables.clone();
    for (i, table) in tables.iter().enumerate() {
        let id = TableIndex::from(i);
        println!("Table {:?}: {:?}", id, table.page_type);
        if matches!(table.page_type, PageType::Unknown(_)) && !parse_unknown_tables {
            println!(
                "  Skipping unknown table type, use --parse-unknown-tables to attempt parsing"
            );
            continue;
        }
        if let Err(e) = dump_table(&mut db, id) {
            eprintln!("Error dumping table {:?}: {e}", id);
        }
    }

    Ok(())
}

fn dump_setting(path: &Path, setting_type: SettingType) -> rekordcrate::Result<()> {
    let mut reader = File::open(path)?;
    let setting = Setting::read_args(&mut reader, (setting_type,))?;

    println!("{:#04x?}", setting);

    Ok(())
}

fn dump_xml(path: &Path) -> rekordcrate::Result<()> {
    let file = File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let document: Document = quick_xml::de::from_reader(reader).expect("failed to deserialize XML");
    println!("{:#?}", document);

    Ok(())
}

fn export(input_dir: &Path, output_dir: &Path) -> rekordcrate::Result<()> {
    use rekordcrate::pdb::string::DeviceSQLString;
    use rekordcrate::util::FileType;
    use std::collections::HashMap;

    let rekordbox_dir = output_dir.join("PIONEER").join("rekordbox");
    let contents_dir = output_dir.join("Contents");
    std::fs::create_dir_all(&rekordbox_dir)?;
    std::fs::create_dir_all(&contents_dir)?;

    // Scan for MP3 files recursively.
    fn find_mp3s(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                find_mp3s(&path, files)?;
            } else if path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("mp3"))
            {
                files.push(path);
            }
        }
        Ok(())
    }

    let mut mp3_files = Vec::new();
    find_mp3s(input_dir, &mut mp3_files)?;
    mp3_files.sort();

    if mp3_files.is_empty() {
        eprintln!("No MP3 files found in {}", input_dir.display());
        return Ok(());
    }

    println!("Found {} MP3 file(s)", mp3_files.len());

    // Collect unique artists and albums, and assign IDs.
    let mut artist_map: HashMap<String, u32> = HashMap::new();
    let mut next_artist_id: u32 = 1;
    let mut album_map: HashMap<(String, u32), u32> = HashMap::new();
    let mut next_album_id: u32 = 1;

    struct TrackInfo {
        dest_filename: String,
        title: String,
        artist_name: String,
        album_name: String,
        duration_secs: u16,
        sample_rate: u32,
        bitrate: u32,
        file_size: u32,
        tempo: u32,
    }

    let mut track_infos = Vec::new();
    let mut used_filenames: HashMap<String, u32> = HashMap::new();

    for src_path in &mp3_files {
        // Determine destination filename with dedup.
        let original_stem = src_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let dest_filename = {
            let base = format!("{}.mp3", original_stem);
            let count = used_filenames.entry(base.clone()).or_insert(0);
            *count += 1;
            if *count == 1 {
                base
            } else {
                format!("{}_{}.mp3", original_stem, count)
            }
        };

        // Copy file to Contents/.
        let dest_path = contents_dir.join(&dest_filename);
        std::fs::copy(src_path, &dest_path)?;

        // Read metadata.
        let file_size = std::fs::metadata(&dest_path)?.len() as u32;
        let tagged_file = lofty::read_from_path(&dest_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let tag = tagged_file
            .primary_tag()
            .or_else(|| tagged_file.first_tag());

        let title = tag
            .and_then(|t| t.title().map(|s| s.to_string()))
            .unwrap_or_else(|| original_stem.clone());

        let artist_name = tag
            .and_then(|t| t.artist().map(|s| s.to_string()))
            .unwrap_or_default();

        let album_name = tag
            .and_then(|t| t.album().map(|s| s.to_string()))
            .unwrap_or_default();

        let properties = tagged_file.properties();
        let duration_secs = properties.duration().as_secs() as u16;
        let sample_rate = properties.sample_rate().unwrap_or(44100);
        let bitrate = properties.overall_bitrate().unwrap_or(320);

        // Analyze BPM.
        let tempo = match bpm_finder_tools::file::analyze_path(&dest_path, 70.0, 200.0) {
            Ok(analysis) => (analysis.bpm * 100.0) as u32,
            Err(e) => {
                eprintln!(
                    "Warning: could not analyze BPM for {}: {e}",
                    dest_path.display()
                );
                0
            }
        };

        // Register artist.
        if !artist_name.is_empty() && !artist_map.contains_key(&artist_name) {
            artist_map.insert(artist_name.clone(), next_artist_id);
            next_artist_id += 1;
        }

        // Register album (keyed by album name + artist ID for uniqueness).
        if !album_name.is_empty() {
            let artist_id = if artist_name.is_empty() {
                0
            } else {
                artist_map[&artist_name]
            };
            let key = (album_name.clone(), artist_id);
            if let std::collections::hash_map::Entry::Vacant(entry) = album_map.entry(key) {
                entry.insert(next_album_id);
                next_album_id += 1;
            }
        }

        track_infos.push(TrackInfo {
            dest_filename,
            title,
            artist_name,
            album_name,
            duration_secs,
            sample_rate,
            bitrate,
            file_size,
            tempo,
        });
    }

    // Create the PDB database.
    // Use the same 20-table layout as real rekordbox exports, including Unknown table types.
    let table_page_types = vec![
        PageType::Plain(PlainPageType::Tracks),
        PageType::Plain(PlainPageType::Genres),
        PageType::Plain(PlainPageType::Artists),
        PageType::Plain(PlainPageType::Albums),
        PageType::Plain(PlainPageType::Labels),
        PageType::Plain(PlainPageType::Keys),
        PageType::Plain(PlainPageType::Colors),
        PageType::Plain(PlainPageType::PlaylistTree),
        PageType::Plain(PlainPageType::PlaylistEntries),
        PageType::Unknown(9),
        PageType::Unknown(10),
        PageType::Plain(PlainPageType::HistoryPlaylists),
        PageType::Plain(PlainPageType::HistoryEntries),
        PageType::Plain(PlainPageType::Artwork),
        PageType::Unknown(14),
        PageType::Unknown(15),
        PageType::Plain(PlainPageType::Columns),
        PageType::Plain(PlainPageType::Menu),
        PageType::Unknown(18),
        PageType::Plain(PlainPageType::History),
    ];

    let pdb_path = rekordbox_dir.join("export.pdb");
    let pdb_file = File::create(&pdb_path)?;
    let mut db = Database::create(pdb_file, DatabaseType::Plain, &table_page_types)?;

    rekordcrate::pdb::defaults::insert_default_colors(&mut db)?;
    rekordcrate::pdb::defaults::insert_default_columns(&mut db)?;
    rekordcrate::pdb::defaults::insert_default_menus(&mut db)?;

    // Insert history sync row with current date.
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    db.add_row(Row::Plain(PlainRow::History(History {
        subtype: Subtype(640),
        index_shift: 0,
        unknown: 0,
        zeroes: 0,
        date: DeviceSQLString::new(&today)?,
        magic: 7705,
        version: DeviceSQLString::new("1000")?,
        label: DeviceSQLString::empty(),
    })))?;

    // Insert artist rows.
    for (name, &id) in &artist_map {
        let artist = Artist::builder()
            .id(id)
            .name(DeviceSQLString::new(name)?)
            .build();
        db.add_row(Row::Plain(PlainRow::Artist(artist)))?;
    }

    // Insert album rows.
    for ((album_name, artist_id), &id) in &album_map {
        let album = Album::builder()
            .id(id)
            .artist_id(*artist_id)
            .name(DeviceSQLString::new(album_name)?)
            .build();
        db.add_row(Row::Plain(PlainRow::Album(album)))?;
    }

    // Insert track rows.
    for (i, info) in track_infos.iter().enumerate() {
        let track_id = (i + 1) as u32;
        let artist_id = if info.artist_name.is_empty() {
            0
        } else {
            artist_map[&info.artist_name]
        };
        let album_id = if info.album_name.is_empty() {
            0
        } else {
            album_map[&(info.album_name.clone(), artist_id)]
        };

        // Pioneer uses forward-slash paths relative to the USB root.
        let pioneer_path = format!("/Contents/{}", info.dest_filename);

        let today = chrono::Local::now().format("%Y-%m-%d").to_string();

        let track = Track::builder()
            .id(track_id)
            .title(DeviceSQLString::new(&info.title)?)
            .artist_id(artist_id)
            .album_id(album_id)
            .file_path(DeviceSQLString::new(&pioneer_path)?)
            .filename(DeviceSQLString::new(&info.dest_filename)?)
            .sample_rate(info.sample_rate)
            .sample_depth(16)
            .bitrate(info.bitrate)
            .duration(info.duration_secs)
            .file_size(info.file_size)
            .file_type(FileType::Mp3)
            .tempo(info.tempo)
            .autoload_hotcues(DeviceSQLString::new("ON")?)
            .date_added(DeviceSQLString::new(&today)?)
            .build();
        db.add_row(Row::Plain(PlainRow::Track(track)))?;

        println!(
            "  [{}] {} - {}{}",
            track_id,
            if info.artist_name.is_empty() {
                "(unknown)"
            } else {
                &info.artist_name
            },
            info.title,
            if info.album_name.is_empty() {
                String::new()
            } else {
                format!(" [{}]", info.album_name)
            },
        );
    }

    // Write the database.
    db.close()?;

    println!(
        "\nExported {} track(s), {} artist(s), and {} album(s) to {}",
        track_infos.len(),
        artist_map.len(),
        album_map.len(),
        pdb_path.display(),
    );

    Ok(())
}

fn guess_db_type(path: &Path, db_type: Option<&str>) -> Option<DatabaseType> {
    let db_type_cli = db_type.map(|str| match str {
        "plain" => DatabaseType::Plain,
        "ext" => DatabaseType::Ext,
        invalid => unreachable!("invalid flag {invalid}, should have already been checked by clap"),
    });
    let file_name = match path.file_name() {
        None => {
            eprintln!("{} not a file!", path.display());
            return None; // TODO(Swiftb0y): turn this into a proper error
        }
        Some(file_name) => file_name,
    };
    let db_type_file = if file_name == "export.pdb" {
        Some(DatabaseType::Plain)
    } else if file_name == "exportExt.pdb" {
        Some(DatabaseType::Ext)
    } else {
        None
    };
    let db_type = match (db_type_cli, db_type_file) {
        (None, None) => {
            eprintln!("no DB_TYPE supplied nor could it be guessed!");
            return None; // TODO(Swiftb0y): turn this into a proper error
        }
        (None, Some(guess)) | (Some(guess), None) => guess,
        (Some(db_type_cli), Some(db_type_file)) if db_type_cli == db_type_file => db_type_cli,
        (Some(db_type_cli), Some(db_type_file)) => {
            eprintln!("Warning: passed {db_type_cli:?}, but found {db_type_file:?} from file name, using {db_type_cli:?}!");
            db_type_cli
        }
    };
    Some(db_type)
}

fn guess_setting_type(path: &Path, setting_type: Option<&str>) -> Option<SettingType> {
    let setting_type_cli = setting_type.map(|str| match str {
        "devsetting" => SettingType::DevSetting,
        "djmmysetting" => SettingType::DJMMySetting,
        "mysetting" => SettingType::MySetting,
        "mysetting2" => SettingType::MySetting2,
        invalid => {
            unreachable!("invalid flag {invalid}, should have already been checked by clap")
        }
    });
    let file_name = match path.file_name() {
        None => {
            eprintln!("{} not a file!", path.display());
            return None; // TODO: turn into proper error
        }
        Some(file_name) => file_name,
    };
    let setting_type_file = SettingType::from_filename(file_name);
    let setting_type = match (setting_type_cli, setting_type_file) {
        (None, None) => {
            eprintln!("no SETTING_TYPE supplied nor could it be guessed!");
            return None; // TODO: turn into proper error
        }
        (None, Some(guess)) | (Some(guess), None) => guess,
        (Some(setting_type_cli), Some(setting_type_file))
            if setting_type_cli == setting_type_file =>
        {
            setting_type_cli
        }
        (Some(setting_type_cli), Some(setting_type_file)) => {
            eprintln!("Warning: passed {setting_type_cli:?}, but found {setting_type_file:?} from file name, using {setting_type_cli:?}!");
            setting_type_cli
        }
    };
    Some(setting_type)
}

fn main() -> rekordcrate::Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::ListPlaylists { path } => list_playlists(path),
        Commands::ListSettings { path } => list_settings(path),
        Commands::ExportPlaylists { path, output_dir } => export_playlists(path, output_dir),
        Commands::DumpPDB {
            path,
            db_type,
            parse_unknown_tables,
        } => {
            let db_type = match guess_db_type(path, db_type.as_deref()) {
                Some(db_type) => db_type,
                None => return Ok(()), // TODO(Swiftb0y): turn into proper error;
            };
            dump_pdb(path, db_type, *parse_unknown_tables)
        }
        Commands::DumpANLZ { path } => dump_anlz(path),
        Commands::DumpSetting { path, setting_type } => {
            let setting_type = match guess_setting_type(path, setting_type.as_deref()) {
                Some(setting_type) => setting_type,
                None => return Ok(()), // TODO: turn into proper error
            };
            dump_setting(path, setting_type)
        }
        Commands::DumpXML { path } => dump_xml(path),
        Commands::Export {
            input_dir,
            output_dir,
        } => export(input_dir, output_dir),
    }
}
